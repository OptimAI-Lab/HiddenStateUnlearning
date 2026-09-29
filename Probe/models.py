import torch
import torch.nn as nn
import copy
from transformers.modeling_outputs import CausalLMOutputWithPast
from accelerate.hooks import remove_hook_from_module

class TrainableDecoder(nn.Module):
    def __init__(self, target_model, device_id=0):
        super().__init__()
        # Dynamically assign to the specified GPU
        self.decoder_device = torch.device(
            f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
        )
        norm    = copy.deepcopy(target_model.model.norm)
        lm_head = copy.deepcopy(target_model.lm_head)

        remove_hook_from_module(norm,    recurse=True)
        remove_hook_from_module(lm_head, recurse=True)

        self.norm    = norm.float().to(self.decoder_device)
        self.lm_head = lm_head.float().to(self.decoder_device)

    def forward(self, hidden):
        hidden = hidden.to(self.decoder_device).float()
        return self.lm_head(self.norm(hidden))

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

def get_hidden_state_at_layer(model, input_ids, layer_idx):
    with torch.no_grad():
        embed_device = next(model.model.embed_tokens.parameters()).device
        hidden = model.model.embed_tokens(input_ids.to(embed_device))

        for i, layer in enumerate(model.model.layers):
            if i > layer_idx:
                break
            layer_device = next(layer.parameters()).device
            hidden = hidden.to(layer_device)
            position_ids = torch.arange(
                hidden.shape[1], device=layer_device
            ).unsqueeze(0)
            hidden = layer(
                hidden, position_ids=position_ids, use_cache=False
            )[0]
    return hidden.float()

class HybridWithTrainedDecoder(nn.Module):
    def __init__(self, backbone_model, decoder: TrainableDecoder, split_layer):
        super().__init__()
        self.backbone    = backbone_model
        self.decoder     = decoder
        self.split_layer = split_layer
        self.config      = backbone_model.config
        self.device      = next(backbone_model.parameters()).device

    def forward(self, input_ids, **kwargs):
        hidden = get_hidden_state_at_layer(
            self.backbone, input_ids, self.split_layer
        )
        return CausalLMOutputWithPast(logits=self.decoder(hidden))

    def generate(self, input_ids, max_new_tokens=32, do_sample=False, pad_token_id=None, **kwargs):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            out        = self.forward(generated)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated  = torch.cat([generated, next_token], dim=1)
            if pad_token_id is not None and next_token.item() == pad_token_id:
                break
        return generated

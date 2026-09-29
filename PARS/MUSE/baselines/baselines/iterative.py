# unlearn_minimax.py

# from .utils import load_model_and_tokenizer, load_model
# from .dataset import ForgetRetainDataset

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.cuda import device_count
# import transformers
# from transformers import Trainer, AutoModelForCausalLM
# import copy
# from accelerate.hooks import remove_hook_from_module
# from typing import List, Optional

# class ProbeDecoder(nn.Module):
#     def __init__(self, source_model):
#         super().__init__()
#         self.decoder_device = torch.device(
#             "cuda:0" if torch.cuda.is_available() else "cpu"
#         )
#         norm    = copy.deepcopy(source_model.model.norm)
#         lm_head = copy.deepcopy(source_model.lm_head)
#         remove_hook_from_module(norm,    recurse=True)
#         remove_hook_from_module(lm_head, recurse=True)
#         self.norm    = norm.float().to(self.decoder_device)
#         self.lm_head = lm_head.float().to(self.decoder_device)

#     def forward(self, hidden):
#         hidden = hidden.to(self.decoder_device).float()
#         return self.lm_head(self.norm(hidden))


# # ── Hidden States WITH gradient (for model outer loop) ────────────────────────

# def get_hidden_state_with_grad(model, input_ids, layer_idx):
    
#     embed_device = next(model.model.embed_tokens.parameters()).device
#     hidden = model.model.embed_tokens(input_ids.to(embed_device))
#     for i, layer in enumerate(model.model.layers):
#         if i > layer_idx:
#             break
#         layer_device = next(layer.parameters()).device
#         hidden = hidden.to(layer_device)
#         position_ids = torch.arange(
#             hidden.shape[1], device=layer_device
#         ).unsqueeze(0)
#         hidden = layer(
#             hidden, position_ids=position_ids, use_cache=False
#         )[0]
#     return hidden.float()


# def get_hidden_state_no_grad(model, input_ids, layer_idx):
    
#     with torch.no_grad():
#         return get_hidden_state_with_grad(model, input_ids, layer_idx)


# def compute_probe_loss(
#     probe:     ProbeDecoder,
#     hidden:    torch.Tensor,   # [B, seq, H]
#     labels:    torch.Tensor,   # [B, seq]  (-100 for prompt tokens)
# ) -> torch.Tensor:
    
#     logits = probe(hidden)                              # [B, seq, vocab]
#     labels = labels.to(probe.decoder_device)

#     shift_logits = logits[..., :-1, :].contiguous()   # [B, seq-1, vocab]
#     shift_labels = labels[..., 1:].contiguous()        # [B, seq-1]

#     loss = nn.CrossEntropyLoss(ignore_index=-100)(
#         shift_logits.view(-1, shift_logits.size(-1)),
#         shift_labels.view(-1),
#     )
#     return loss


# def unlearn_minimax(
#     model_dir:              str,
#     data_file:              str,
#     out_dir:                str,
#     retain_data_file:       Optional[str]  = None,
#     loss_type:              str            = 'npo',
#     per_device_batch_size:  int            = 2,
#     epochs:                 int            = 5,
#     learning_rate:          float          = 1e-5,
#     max_len:                int            = 4096,
#     tokenizer_dir:          Optional[str]  = None,
#     resume_from_checkpoint: bool           = False,
#     # Minimax probe arguments
#     probe_layers:           List[int]      = None,   # e.g. [8, 10, 12, 14]
#     probe_lr:               float          = 1e-4,   # decoder trains faster
#     probe_inner_steps:      int            = 3,      # k inner steps per outer
#     alpha:                  float          = 1.0,    # retain loss weight
#     beta:                   float          = 0.5,    # probe loss weight
# ):
    
#     if 'gd' in loss_type:
#         assert retain_data_file is not None, \
#             "Retain data must be specified for grad_diff."

#     if probe_layers is None:
#         # Default: layers identified as high-leakage from probing experiments
#         # For verbmem: layers 8-14; for knowmem: layers 28-31
#         probe_layers = [8, 10, 12, 14]

#     from .utils import load_model_and_tokenizer, load_model
#     model, tokenizer = load_model_and_tokenizer(
#         model_dir, tokenizer_dir=tokenizer_dir
#     )

#     ref_model = (
#         load_model(model_dir)
#         if 'npo' in loss_type or 'kl' in loss_type
#         else None
#     )

#     dataset = ForgetRetainDataset(
#         data_file,
#         tokenizer=tokenizer,
#         retain_file_path=retain_data_file,
#         max_len=max_len,
#     )

#     if device_count() == 0:
#         raise ValueError("Device not detected!")

#     training_args = transformers.TrainingArguments(
#         output_dir=out_dir,
#         per_device_train_batch_size=per_device_batch_size,
#         learning_rate=learning_rate,
#         save_strategy='epoch',
#         num_train_epochs=epochs,
#         optim='adamw_torch',
#         lr_scheduler_type='constant',
#         bf16=True,
#         report_to='none',
#     )

#     trainer = MinimaxUnlearner(
#         model=model,
#         ref_model=ref_model,
#         tokenizer=tokenizer,
#         train_dataset=dataset,
#         args=training_args,
#         data_collator=dataset.get_collate_fn(),
#         loss_type=loss_type,
#         probe_layers=probe_layers,
#         probe_lr=probe_lr,
#         probe_inner_steps=probe_inner_steps,
#         alpha=alpha,
#         beta=beta,
#     )

#     model.config.use_cache = False
#     trainer.train(resume_from_checkpoint=resume_from_checkpoint)
#     trainer.save_model(out_dir)


# class MinimaxUnlearner(Trainer):

#     def __init__(
#         self,
#         *args,
#         loss_type:         str        = 'npo',
#         ref_model:         Optional[AutoModelForCausalLM] = None,
#         beta_npo:          float      = 0.1,
#         probe_layers:      List[int]  = None,
#         probe_lr:          float      = 1e-4,
#         probe_inner_steps: int        = 3,
#         alpha:             float      = 1.0,
#         beta:              float      = 0.5,
#         **kwargs,
#     ):
#         self.loss_type         = loss_type
#         self.ref_model         = ref_model
#         self.beta_npo          = beta_npo       # NPO temperature
#         self.probe_layers      = probe_layers or [8, 10, 12, 14]
#         self.probe_lr          = probe_lr
#         self.probe_inner_steps = probe_inner_steps
#         self.alpha             = alpha          # retain weight
#         self.beta              = beta           # probe weight

#         if ref_model is not None:
#             assert 'po' in loss_type or 'kl' in loss_type
#             ref_model.eval()

#         super().__init__(*args, **kwargs)

#         self.probes = {
#             ell: ProbeDecoder(self.model)
#             for ell in self.probe_layers
#         }
        
#         self.probe_optimizers = {
#             ell: torch.optim.AdamW(
#                 self.probes[ell].parameters(), lr=self.probe_lr
#             )
#             for ell in self.probe_layers
#         }

#     def compute_loss(self, model, x, return_outputs=False):
#         x_f, x_r = x

#         # ── Standard output-level forward passes ──────────────────────────
#         outputs_f = model(
#             x_f['input_ids'],
#             labels=x_f.get('labels', x_f['input_ids'].clone()),
#             attention_mask=x_f.get(
#                 'attention_mask',
#                 torch.ones_like(x_f['input_ids'], dtype=torch.bool)
#             ),
#         )
#         loss_f = outputs_f.loss

#         if 'gdr' in self.loss_type or 'klr' in self.loss_type:
#             outputs_r = model(
#                 x_r['input_ids'],
#                 labels=x_r.get('labels', x_r['input_ids'].clone()),
#                 attention_mask=x_r.get(
#                     'attention_mask',
#                     torch.ones_like(x_r['input_ids'], dtype=torch.bool)
#                 ),
#             )
#             loss_r = outputs_r.loss

#         if 'npo' in self.loss_type:
#             with torch.no_grad():
#                 outputs_f_ref = self.ref_model(
#                     x_f['input_ids'],
#                     labels=x_f.get('labels', x_f['input_ids'].clone()),
#                     attention_mask=x_f.get(
#                         'attention_mask',
#                         torch.ones_like(x_f['input_ids'], dtype=torch.bool)
#                     ),
#                 )

#         forget_labels = x_f.get('labels', x_f['input_ids'].clone())
#         forget_ids    = x_f['input_ids']
#         embed_device  = next(model.model.embed_tokens.parameters()).device

#         for p in model.parameters():
#             p.requires_grad_(False)

#         for _ in range(self.probe_inner_steps):
#             for ell in self.probe_layers:
#                 probe = self.probes[ell]
#                 probe.train()
#                 self.probe_optimizers[ell].zero_grad()
                
#                 hidden = get_hidden_state_no_grad(
#                     model, forget_ids, ell
#                 )  

#                 loss_probe_inner = compute_probe_loss(
#                     probe, hidden, forget_labels
#                 )

#                 loss_probe_inner.backward()
#                 nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
#                 self.probe_optimizers[ell].step()

#         for p in model.parameters():
#             p.requires_grad_(True)

#         # Freeze probe decoders during outer loop
#         for ell in self.probe_layers:
#             for p in self.probes[ell].parameters():
#                 p.requires_grad_(False)

#         loss = torch.tensor(0.0, device=embed_device)

#         if 'ga' in self.loss_type:
#             loss = loss + (-loss_f)

#         elif 'npo' in self.loss_type:
#             neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
#             loss = loss + (
#                 -F.logsigmoid(self.beta_npo * neg_log_ratio).mean()
#                 * 2 / self.beta_npo
#             )
#         else:
#             raise NotImplementedError(
#                 f"Loss type '{self.loss_type}' not recognized."
#             )

#         if 'gdr' in self.loss_type:
#             loss = loss + self.alpha * loss_r

#         if 'klr' in self.loss_type:
#             with torch.no_grad():
#                 outputs_r_ref = self.ref_model(
#                     x_r['input_ids'],
#                     labels=x_r.get('labels', x_r['input_ids'].clone()),
#                     attention_mask=x_r.get(
#                         'attention_mask',
#                         torch.ones_like(x_r['input_ids'], dtype=torch.bool)
#                     ),
#                 )
#             kl_r = F.kl_div(
#                 outputs_r.logits,
#                 outputs_r_ref.logits,
#                 reduction='batchmean',
#                 log_target=True,
#             )
#             loss = loss + self.alpha * kl_r

#         loss_probe_outer = torch.tensor(0.0, device=embed_device)
#         for ell in self.probe_layers:
#             probe = self.probes[ell]
#             probe.eval()

#             # Hidden states WITH grad — gradient will flow back to theta
#             hidden = get_hidden_state_with_grad(
#                 model, forget_ids, ell
#             )   # [B, seq, H], grad_fn attached

#             loss_probe_outer = loss_probe_outer + compute_probe_loss(
#                 probe, hidden, forget_labels
#             )

#         loss_probe_outer = loss_probe_outer / len(self.probe_layers)
#         loss = loss + self.beta * loss_probe_outer

#         # Re-enable probe gradients for next step's inner loop
#         for ell in self.probe_layers:
#             for p in self.probes[ell].parameters():
#                 p.requires_grad_(True)

#         return (loss, outputs_f) if return_outputs else loss

#     def prediction_step(
#         self, model, x, prediction_loss_only: bool, ignore_keys=None
#     ):
#         input_ids, labels, attention_mask = x
#         with torch.no_grad():
#             outputs = model(
#                 input_ids, labels=labels, attention_mask=attention_mask
#             )
#         return (outputs.loss, outputs.logits, labels)

from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
import copy
from accelerate.hooks import remove_hook_from_module
from typing import List, Optional

class ProbeDecoder(nn.Module):
    
    def __init__(self, source_model, probe_device: int = 7):
        super().__init__()
        self.decoder_device = torch.device(
            f"cuda:{probe_device}" if torch.cuda.is_available() else "cpu"
        )
        norm    = copy.deepcopy(source_model.model.norm)
        lm_head = copy.deepcopy(source_model.lm_head)
        remove_hook_from_module(norm,    recurse=True)
        remove_hook_from_module(lm_head, recurse=True)
        self.norm    = norm.float().to(self.decoder_device)
        self.lm_head = lm_head.float().to(self.decoder_device)

    def forward(self, hidden):
        hidden = hidden.to(self.decoder_device).float()
        return self.lm_head(self.norm(hidden))


def get_hidden_state_with_grad(model, input_ids, layer_idx):
    
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
    # Return hidden with gradient support (outer loop needs grad)
    return hidden.float()


def get_hidden_state_no_grad(model, input_ids, layer_idx):
    with torch.no_grad():
        h = get_hidden_state_with_grad(model, input_ids, layer_idx)
        return h.detach()


def compute_probe_loss(
    probe:   ProbeDecoder,
    hidden:  torch.Tensor,   
    labels:  torch.Tensor,   
) -> torch.Tensor:
    
    logits = probe(hidden)                              
    labels = labels.to(probe.decoder_device)

    shift_logits = logits[..., :-1, :].contiguous()    
    shift_labels = labels[..., 1:].contiguous()         

    loss = nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )
    return loss


def unlearn_minimax(
    model_dir:              str,
    data_file:              str,
    out_dir:                str,
    retain_data_file:       str | None  = None,
    loss_type:              str         = 'ga',
    per_device_batch_size:  int         = 1,
    epochs:                 int         = 5,
    learning_rate:          float       = 1e-5,
    max_len:                int         = 1024,
    tokenizer_dir:          str | None  = None,
    resume_from_checkpoint: bool        = False,
    # Minimax probe arguments
    probe_layers:           List[int]   = None,
    probe_lr:               float       = 1e-4,
    probe_inner_steps:      int         = 3,
    probe_beta:             float       = 0.5,
    probe_device:           int         = 7,  # Probe runs on separate GPU
    keep_checkpoints:       int         = 1,
):
    print("loss type: ",loss_type)
    print("batch size: ",per_device_batch_size)
    if 'gd' in loss_type:
        assert retain_data_file is not None, \
            "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir,
    )

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type
        else None
    )

    dataset = ForgetRetainDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len,
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='no',
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none',
        gradient_accumulation_steps=8,  # Effective batch size = 8
        gradient_checkpointing=True,    # Enable gradient checkpointing
    )

    trainer = MinimaxUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        probe_layers=probe_layers or [],
        probe_lr=probe_lr,
        probe_inner_steps=probe_inner_steps,
        probe_beta=probe_beta,
        probe_device=probe_device,
    )

    model.config.use_cache = False
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)


class MinimaxUnlearner(Trainer):
    
    def __init__(
        self,
        *args,
        loss_type:         str             = 'ga',
        ref_model:         AutoModelForCausalLM | None = None,
        beta:              float           = 0.1,
        # Probe arguments
        probe_layers:      List[int]       = None,
        probe_lr:          float           = 1e-4,
        probe_inner_steps: int             = 3,
        probe_beta:        float           = 0.5,
        probe_device:      int             = 7,  # Probe runs on separate GPU
        **kwargs,
    ):
        self.loss_type         = loss_type
        self.ref_model         = ref_model
        self.beta              = beta              # NPO temperature
        self.probe_layers      = probe_layers or []
        self.probe_lr          = probe_lr
        self.probe_inner_steps = probe_inner_steps
        self.probe_beta        = probe_beta        # weight of probe term
        self.probe_device      = probe_device      # Probe GPU device

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)

        if self.probe_layers:
            self.probes = {
                ell: ProbeDecoder(self.model, probe_device=self.probe_device)
                for ell in self.probe_layers
            }
            
            self.probe_optimizers = {
                ell: torch.optim.AdamW(
                    self.probes[ell].parameters(), lr=self.probe_lr
                )
                for ell in self.probe_layers
            }
        else:
            self.probes           = {}
            self.probe_optimizers = {}

    def compute_loss(self, model, x, return_outputs=False):

        x_f, x_r = x

        forget_ids    = x_f['input_ids']
        forget_labels = x_f['labels'] if 'labels' in x_f \
                        else x_f['input_ids'].clone()
        forget_mask   = x_f['attention_mask'] if 'attention_mask' in x_f \
                        else torch.ones_like(x_f['input_ids'], dtype=torch.bool)

        outputs_f = model(
            forget_ids,
            labels=forget_labels,
            attention_mask=forget_mask,
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            retain_ids    = x_r['input_ids']
            retain_labels = x_r['labels'] if 'labels' in x_r \
                            else x_r['input_ids'].clone()
            retain_mask   = x_r['attention_mask'] if 'attention_mask' in x_r \
                            else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            outputs_r = model(
                retain_ids,
                labels=retain_labels,
                attention_mask=retain_mask,
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' in self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    forget_ids,
                    labels=forget_labels,
                    attention_mask=forget_mask,
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    retain_ids,
                    labels=retain_labels,
                    attention_mask=retain_mask,
                )

        if self.probe_layers:
            for p in model.parameters():
                p.requires_grad_(False)

            for _ in range(self.probe_inner_steps):
                for ell in self.probe_layers:
                    probe = self.probes[ell]
                    probe.train()
                    self.probe_optimizers[ell].zero_grad()

                    hidden = get_hidden_state_no_grad(
                        model, forget_ids, ell
                    )   
                    loss_probe_inner = compute_probe_loss(
                        probe, hidden, forget_labels
                    )
                    loss_probe_inner.backward()
                    nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
                    self.probe_optimizers[ell].step()

            for p in model.parameters():
                p.requires_grad_(True)

            for ell in self.probe_layers:
                for p in self.probes[ell].parameters():
                    p.requires_grad_(False)

        # Initialize loss on the same device as model outputs to avoid device mismatch
        loss = loss_f.new_zeros(())

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' in self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(
                self.beta * neg_log_ratio
            ).mean() * 2 / self.beta

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            loss += loss_r

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction='batchmean',
                log_target=True,
            )
            loss += kl_r

        if self.probe_layers:
            # Accumulate probe loss on the same device as `loss` to avoid cross-device
            # addition; cast probe losses to `loss` device when adding.
            loss_probe_outer = loss.new_zeros(())
            for ell in self.probe_layers:
                probe = self.probes[ell]
                probe.eval()

                hidden = get_hidden_state_with_grad(
                    model, forget_ids, ell
                )

                lp = compute_probe_loss(probe, hidden, forget_labels)
                loss_probe_outer = loss_probe_outer + lp.to(loss_probe_outer.device)

            loss_probe_outer = loss_probe_outer / len(self.probe_layers)
            loss = loss - self.probe_beta * loss_probe_outer

            for ell in self.probe_layers:
                for p in self.probes[ell].parameters():
                    p.requires_grad_(True)

        return (loss, outputs_f) if return_outputs else loss

    def prediction_step(
        self, model, x, prediction_loss_only: bool, ignore_keys=None
    ):
        input_ids, labels, attention_mask = x
        with torch.no_grad():
            outputs = model(
                input_ids, labels=labels, attention_mask=attention_mask
            )
            logits = outputs.logits
            loss   = outputs.loss
        return (loss, logits, labels)
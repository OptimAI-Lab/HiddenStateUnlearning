import os
import json
import csv
import gc
import torch
import torch.nn as nn
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from tqdm import tqdm
from tqdm.contrib import tzip
from typing import List, Tuple, Optional, Dict
from rouge_score import rouge_scorer
from accelerate.hooks import remove_hook_from_module

DEFAULT_DATA = {
    'news': {
        'verbmem_forget_file': "data/news/verbmem/forget.json",
    },
}


def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def write_csv(rows: List[dict], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class RougeEvalLogger:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.scores = []
        self.logs   = []

    def log(self, prompt, gt, output, question=None):
        score = self.scorer.score(gt, output)['rougeL'].fmeasure
        self.scores.append(score)
        self.logs.append({
            "prompt": prompt,
            "gt":     gt,
            "output": output,
            "rougeL": score,
        })

    def report(self):
        mean = sum(self.scores) / len(self.scores) if self.scores else 0.0
        return {"rougeL_mean": mean, "rougeL_all": self.scores, "logs": self.logs}


def eval_verbmem(
    model,
    tokenizer,
    prompts:        List[str],
    gts:            List[str],
    max_new_tokens: int = 128,
):
    
    logger = RougeEvalLogger()

    for prompt, gt in tzip(prompts, gts):
        input_ids = tokenizer(
            prompt, return_tensors='pt', add_special_tokens=True
        ).input_ids

        gt_ids = tokenizer(
            gt, return_tensors='pt', add_special_tokens=True
        ).input_ids[:, :max_new_tokens]

        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        output_ids = output_ids[:, len(input_ids[0]):]

        output = tokenizer.batch_decode(
            output_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        gt_short = tokenizer.batch_decode(
            gt_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        logger.log(prompt, gt_short, output)

    agg = logger.report()
    return agg, agg['logs']


class TrainableDecoder(nn.Module):
    
    def __init__(self, source_model):
        super().__init__()
        self.decoder_device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
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

    def generate(self, input_ids, max_new_tokens=128,
                 do_sample=False, pad_token_id=None, **kwargs):
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            out        = self.forward(generated)
            next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated  = torch.cat([generated, next_token], dim=1)
            if pad_token_id is not None and \
               next_token.item() == pad_token_id:
                break
        return generated


def build_verbmem_samples(
    tokenizer,
    prompts:        List[str],
    gts:            List[str],
    max_length:     int = 2048,
    max_new_tokens: int = 128,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[str, str]]]:
    
    samples     = []
    valid_pairs = []

    for prompt, gt in zip(prompts, gts):
        prompt_ids = tokenizer(
            prompt, return_tensors='pt', add_special_tokens=True,
            truncation=True, max_length=max_length,
        ).input_ids[0]

        gt_ids = tokenizer(
            gt, return_tensors='pt', add_special_tokens=False,
            truncation=True, max_length=max_new_tokens,
        ).input_ids[0]

        full_ids = torch.cat([prompt_ids, gt_ids], dim=0)[:max_length]
        q_len    = prompt_ids.shape[0]

        if full_ids.shape[0] - q_len <= 0:
            continue

        samples.append((full_ids, q_len))
        valid_pairs.append((prompt, gt))

    return samples, valid_pairs


def compute_batch_loss_verbmem(
    decoder, probe_model, samples, split_layer,
    batch_size, tokenizer, no_grad=True,
) -> Optional[float]:
    embed_device = next(probe_model.model.embed_tokens.parameters()).device
    total_loss   = 0.0
    n_batches    = 0
    ctx          = torch.no_grad() if no_grad else torch.enable_grad()

    with ctx:
        for i in range(0, len(samples), batch_size):
            batch   = samples[i:i+batch_size]
            max_len = max(s[0].shape[0] for s in batch)
            pad_id  = tokenizer.pad_token_id

            batch_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long
            )
            labels = torch.full(
                (len(batch), max_len), -100, dtype=torch.long
            )
            for b, (ids, q_len) in enumerate(batch):
                seq_len = ids.shape[0]
                batch_ids[b, :seq_len] = ids
                labels[b, q_len:seq_len] = ids[q_len:seq_len]

            batch_ids = batch_ids.to(embed_device)
            labels    = labels.to(decoder.decoder_device)

            if (labels != -100).sum() == 0:
                continue

            hidden = get_hidden_state_at_layer(
                probe_model, batch_ids, split_layer
            )
            logits = decoder(hidden)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else None


def train_verbmem_decoder_on_model(
    probe_model,
    tokenizer,
    prompts:        List[str],
    gts:            List[str],
    split_layer:    int,
    max_epochs:     int   = 50,
    batch_size:     int   = 4,
    lr:             float = 5e-5,
    max_length:     int   = 2048,
    max_new_tokens: int   = 128,
    val_ratio:      float = 0.2,
    patience:       int   = 3,
    min_epochs:     int   = 3,
    seed:           int   = 42,
) -> Tuple[TrainableDecoder, dict, List[str], List[str]]:
    
    embed_device = next(probe_model.model.embed_tokens.parameters()).device
    torch.manual_seed(seed)

    all_samples, valid_pairs = build_verbmem_samples(
        tokenizer, prompts, gts, max_length, max_new_tokens
    )
    N = len(all_samples)

    if N == 0:
        print(f"  ERROR: no valid verbmem samples at layer {split_layer}")
        return TrainableDecoder(probe_model), {}, [], []

    n_val   = max(1, int(N * val_ratio))
    n_train = N - n_val
    perm    = torch.randperm(
        N, generator=torch.Generator().manual_seed(seed)
    ).tolist()

    train_samples = [all_samples[i]    for i in perm[:n_train]]
    val_samples   = [all_samples[i]    for i in perm[n_train:]]
    val_prompts   = [valid_pairs[i][0] for i in perm[n_train:]]
    val_gts       = [valid_pairs[i][1] for i in perm[n_train:]]

    print(f"  [VerbMem] {n_train} train / {n_val} val  "
          f"(layer={split_layer}  max_epochs={max_epochs}  patience={patience})")

    decoder   = TrainableDecoder(probe_model)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)

    test_ids = train_samples[0][0].unsqueeze(0).to(embed_device)
    with torch.no_grad():
        h_test  = get_hidden_state_at_layer(probe_model, test_ids, split_layer)
        lg_test = decoder(h_test)
    print(f"  Diagnostic: hidden_max={h_test.abs().max():.1f}  "
          f"logit_max={lg_test.abs().max():.1f}  "
          f"nan={lg_test.isnan().any().item()}")

    best_val_loss   = float('inf')
    best_state_dict = copy.deepcopy(decoder.state_dict())
    best_epoch      = 0
    patience_count  = 0
    history         = []

    decoder.train()

    for epoch in range(max_epochs):
        epoch_loss  = 0.0
        n_batches   = 0
        nan_batches = 0
        perm_train  = torch.randperm(n_train).tolist()

        for i in range(0, n_train, batch_size):
            batch   = [train_samples[j] for j in perm_train[i:i+batch_size]]
            max_len = max(s[0].shape[0] for s in batch)
            pad_id  = tokenizer.pad_token_id

            batch_ids = torch.full(
                (len(batch), max_len), pad_id, dtype=torch.long
            )
            labels = torch.full(
                (len(batch), max_len), -100, dtype=torch.long
            )
            for b, (ids, q_len) in enumerate(batch):
                seq_len = ids.shape[0]
                batch_ids[b, :seq_len] = ids
                labels[b, q_len:seq_len] = ids[q_len:seq_len]

            batch_ids = batch_ids.to(embed_device)
            labels    = labels.to(decoder.decoder_device)

            if (labels != -100).sum() == 0:
                nan_batches += 1
                continue

            hidden = get_hidden_state_at_layer(
                probe_model, batch_ids, split_layer
            )
            logits = decoder(hidden)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        if n_batches == 0:
            print(f"  Epoch {epoch+1:>3}  ALL NaN/empty")
            continue

        train_loss = epoch_loss / n_batches

        decoder.eval()
        val_loss = compute_batch_loss_verbmem(
            decoder, probe_model, val_samples,
            split_layer, batch_size, tokenizer, no_grad=True
        )
        decoder.train()

        if val_loss is None:
            print(f"  Epoch {epoch+1:>3}  train={train_loss:.4f}  val=NaN")
            continue

        if epoch + 1 >= min_epochs and val_loss < best_val_loss:
            best_val_loss   = val_loss
            best_state_dict = copy.deepcopy(decoder.state_dict())
            best_epoch      = epoch + 1
            patience_count  = 0
            improved        = "best"
        elif epoch + 1 >= min_epochs:
            patience_count += 1
            improved        = f"patience {patience_count}/{patience}"
        else:
            best_val_loss   = val_loss
            best_state_dict = copy.deepcopy(decoder.state_dict())
            best_epoch      = epoch + 1
            improved        = "warmup"

        history.append({
            'epoch':      epoch + 1,
            'train_loss': train_loss,
            'val_loss':   val_loss,
        })

        msg = (f"  Epoch {epoch+1:>3}/{max_epochs}  "
               f"train={train_loss:.4f}  val={val_loss:.4f}  {improved}")
        if nan_batches:
            msg += f"  ({nan_batches} skipped)"
        print(msg)

        if epoch + 1 >= min_epochs and patience_count >= patience:
            print(f"  Early stop  best={best_epoch}  val={best_val_loss:.4f}")
            break

    decoder.load_state_dict(best_state_dict)
    decoder.eval()   # ← decoder is in eval() mode when returned
    print(f"  Best: epoch={best_epoch}  val_loss={best_val_loss:.4f}  "
          f"val_size={len(val_prompts)}")

    info = {
        'best_epoch':    best_epoch,
        'best_val_loss': best_val_loss,
        'history':       history,
        'n_train':       n_train,
        'n_val':         n_val,
    }
    return decoder, info, val_prompts, val_gts


def _build_csv_rows(layer_results, model_names, eval_layers):
    
    rows = []
    for ell in eval_layers:
        for name in model_names:
            if name not in layer_results:
                continue
            if ell not in layer_results[name]:
                continue
            r = layer_results[name][ell]
            rows.append({
                'layer':            ell,
                'model':            name,
                'best_epoch':       r['best_epoch'],
                'best_val_loss':    round(r['best_val_loss'], 4),
                'n_val':            r['n_val'],
                'vm_original_full': round(r['vm_original_full'], 2),
                'vm_own_full':      round(r['vm_own_full'], 2),
                'vm_unlearned_val': round(r['vm_unlearned_val'], 2),
                'vm_dec_val':       round(r['vm_dec_val'], 2),
                'vm_direct_val':    round(r['vm_direct_val'], 2),   # NEW
                'vm_direct_full':   round(r['vm_direct_full'], 2),
                'vs_own':           round(r['vs_own'], 2),
                'vs_original':      round(r['vs_original'], 2),
            })
    return rows


if __name__ == "__main__":

    ORIGINAL_MODEL_PATH = ...
    TOKENIZER_DIR       = ...
    CORPUS              = ...
    TEMP_DIR            = ...

    UNLEARNED_MODEL_PATHS = {
        "BLUR-NPO": ...,
        "NPO":      ...,
        "SimNPO":   ...,
        "SAM":  ...,
        "SURE": ...,
        "PARS": ...,
    }

    MAX_EPOCHS     = 50
    MIN_EPOCHS     = 3
    PATIENCE       = 3
    VAL_RATIO      = 0.2
    LR             = 5e-5
    BATCH_SIZE     = 4
    SEED           = 42
    MAX_NEW_TOKENS = 128

    NUM_LAYERS  = 32
    EVAL_LAYERS = list(range(NUM_LAYERS))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    verbmem_data = read_json(DEFAULT_DATA[CORPUS]['verbmem_forget_file'])
    prompts = [d['prompt'] for d in verbmem_data]
    gts     = [d['gt']     for d in verbmem_data]
    print(f"  VerbMem forget set: {len(prompts)} samples")

    original_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    original_model.eval()

    print("-" * 70)
    print("VerbMem baselines on full forget set")
    print("-" * 70)

    agg, _ = eval_verbmem(
        model=original_model, tokenizer=tokenizer,
        prompts=prompts, gts=gts,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    vm_original_full = agg['rougeL_mean'] * 100
    write_json(agg, os.path.join(TEMP_DIR, "baselines/original_verbmem.json"))
    print(f"  Original: {vm_original_full:.2f}")

    all_layer_results = {}
    all_csv_rows      = []

    for name, path in UNLEARNED_MODEL_PATHS.items():
        print("\n" + "=" * 70)
        print(f"Loading {name}  ({path})...")
        print("=" * 70)

        unlearned_model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype="auto",
            device_map="auto",
        )
        unlearned_model.eval()

        agg, _ = eval_verbmem(
            model=unlearned_model, tokenizer=tokenizer,
            prompts=prompts, gts=gts,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        vm_own_full = agg['rougeL_mean'] * 100
        write_json(agg, os.path.join(
            TEMP_DIR, f"baselines/{name}_verbmem.json"
        ))
        print(f"  {name} verbmem baseline: {vm_own_full:.2f}")

        all_layer_results[name] = {}

        for ell in EVAL_LAYERS:
            print("-" * 70)
            print(f"  [{name}] Layer {ell} / {NUM_LAYERS-1}")
            print("-" * 70)

            decoder, info, val_prompts, val_gts_split = \
                train_verbmem_decoder_on_model(
                    probe_model=unlearned_model,
                    tokenizer=tokenizer,
                    prompts=prompts,
                    gts=gts,
                    split_layer=ell,
                    max_epochs=MAX_EPOCHS,
                    batch_size=BATCH_SIZE,
                    lr=LR,
                    max_new_tokens=MAX_NEW_TOKENS,
                    val_ratio=VAL_RATIO,
                    patience=PATIENCE,
                    min_epochs=MIN_EPOCHS,
                    seed=SEED,
                )

            if not info or len(val_prompts) == 0:
                print(f"  ERROR: skipping layer {ell}")
                continue

            print(f"\n  Baselines on val split ({len(val_prompts)} samples):")

            agg_unl_val, _ = eval_verbmem(
                model=unlearned_model, tokenizer=tokenizer,
                prompts=val_prompts, gts=val_gts_split,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            vm_unlearned_val = agg_unl_val['rougeL_mean'] * 100
            print(f"  {name:<20} val: {vm_unlearned_val:.2f}")

            print(f"\n  Validity: decoder on {name} R^({ell}) [val split]")
            val_hybrid = HybridWithTrainedDecoder(
                backbone_model=unlearned_model,
                decoder=decoder,
                split_layer=ell,
            )
            agg_dec_val, _ = eval_verbmem(
                model=val_hybrid, tokenizer=tokenizer,
                prompts=val_prompts, gts=val_gts_split,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            vm_dec_val = agg_dec_val['rougeL_mean'] * 100
            print(f"  dec_val={vm_dec_val:.2f}  "
                  f"(unlearned_val={vm_unlearned_val:.2f})")
            del val_hybrid

            decoder.freeze()

            print(f"\n  Frozen decoder on {name} R^({ell}) [val split]")
            hybrid_val = HybridWithTrainedDecoder(
                backbone_model=unlearned_model,
                decoder=decoder,
                split_layer=ell,
            )
            agg_direct_val, _ = eval_verbmem(
                model=hybrid_val, tokenizer=tokenizer,
                prompts=val_prompts, gts=val_gts_split,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            vm_direct_val = agg_direct_val['rougeL_mean'] * 100
            print(f"  direct_val={vm_direct_val:.2f}  "
                  f"(unlearned_val={vm_unlearned_val:.2f})")
            write_json(agg_direct_val, os.path.join(
                TEMP_DIR, name, f"layer_{ell}/verbmem_val_agg.json"
            ))
            del hybrid_val

            print(f"\n  Frozen decoder on {name} R^({ell}) [full set]")
            hybrid_full = HybridWithTrainedDecoder(
                backbone_model=unlearned_model,
                decoder=decoder,
                split_layer=ell,
            )
            agg_full, log_full = eval_verbmem(
                model=hybrid_full, tokenizer=tokenizer,
                prompts=prompts, gts=gts,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            vm_direct_full = agg_full['rougeL_mean'] * 100
            vs             = vm_direct_full - vm_own_full
            vs_o           = vm_direct_full - vm_original_full

            print(f"  direct_full={vm_direct_full:.2f}  "
                  f"vs {name}: {vs:+.2f}  "
                  f"vs original: {vs_o:+.2f}")

            write_json(agg_full,  os.path.join(
                TEMP_DIR, name, f"layer_{ell}/verbmem_agg.json"
            ))
            write_json(log_full,  os.path.join(
                TEMP_DIR, name, f"layer_{ell}/verbmem_log.json"
            ))
            write_json(info, os.path.join(
                TEMP_DIR, name, f"layer_{ell}/verbmem_training_info.json"
            ))
            del hybrid_full

            all_layer_results[name][ell] = {
                'best_epoch':       info['best_epoch'],
                'best_val_loss':    info['best_val_loss'],
                'n_val':            len(val_prompts),
                'vm_original_full': vm_original_full,
                'vm_own_full':      vm_own_full,
                'vm_unlearned_val': vm_unlearned_val,
                'vm_dec_val':       vm_dec_val,       
                'vm_direct_val':    vm_direct_val,   
                'vm_direct_full':   vm_direct_full,   
                'vs_own':           vs,
                'vs_original':      vs_o,
            }

            del decoder
            gc.collect()
            torch.cuda.empty_cache()

            all_csv_rows = _build_csv_rows(
                all_layer_results,
                list(UNLEARNED_MODEL_PATHS.keys()),
                EVAL_LAYERS,
            )
            write_csv(all_csv_rows,
                      os.path.join(TEMP_DIR, "verbmem_results.csv"))

        del unlearned_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n  Original VerbMem: {vm_original_full:.2f}")

    for name in UNLEARNED_MODEL_PATHS:
        if name not in all_layer_results:
            continue
        print(f"\n  [{name}]")
        for ell in sorted(all_layer_results[name].keys()):
            r = all_layer_results[name][ell]
            print(f"    Layer {ell:>2}  "
                  f"dec_val={r['vm_dec_val']:.2f}  "
                  f"direct_val={r['vm_direct_val']:.2f}  "
                  f"direct_full={r['vm_direct_full']:.2f}  "
                  f"vs_own={r['vs_own']:+.2f}  "
                  f"vs_orig={r['vs_original']:+.2f}")
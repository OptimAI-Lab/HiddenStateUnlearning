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
        'knowmem_forget_qa_file':     "data/news/knowmem/forget_qa.json",
        'knowmem_forget_qa_icl_file': "data/news/knowmem/forget_qa_icl.json",
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
        self.scorer   = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.scores_f = []   # F-measure
        self.scores_r = []   # Recall
        self.logs     = []

    def log(self, prompt, answer, output, question=None):
        result    = self.scorer.score(answer, output)['rougeL']
        f_measure = result.fmeasure
        recall    = result.recall
        self.scores_f.append(f_measure)
        self.scores_r.append(recall)
        self.logs.append({
            "question": question,
            "prompt":   prompt,
            "answer":   answer,
            "output":   output,
            "rougeL_f": f_measure,
            "rougeL_r": recall,
        })

    def report(self):
        mean_f = sum(self.scores_f) / len(self.scores_f) if self.scores_f else 0.0
        mean_r = sum(self.scores_r) / len(self.scores_r) if self.scores_r else 0.0
        return {
            "rougeL_f_mean": mean_f,
            "rougeL_r_mean": mean_r,
            "rougeL_f_all":  self.scores_f,
            "rougeL_r_all":  self.scores_r,
            "logs":          self.logs,
        }



def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words:
        string = string.split(word)[0]
    return string


def eval_knowmem(
    model,
    tokenizer,
    questions:      List[str],
    answers:        List[str],
    icl_qs:         List[str] = [],
    icl_as:         List[str] = [],
    max_new_tokens: int       = 32,
):
    logger = RougeEvalLogger()

    general_prompt = ""
    for q, a in zip(icl_qs, icl_as):
        general_prompt += f"Question: {q}\nAnswer: {a}\n\n"

    for question, answer in tzip(questions, answers):
        prompt    = general_prompt + f"Question: {question}\nAnswer: "
        input_ids = tokenizer(
            prompt, return_tensors='pt', add_special_tokens=True
        ).input_ids

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
        output = get_prefix_before_words_occur(
            output, ["\n\n", "\nQuestion", "Question:"]
        )
        logger.log(prompt, answer, output, question=question)

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

    def generate(self, input_ids, max_new_tokens=32,
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


def build_samples(
    tokenizer,
    questions:  List[str],
    answers:    List[str],
    max_length: int = 2048,
) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[str, str]]]:
    samples     = []
    valid_pairs = []
    for q, a in zip(questions, answers):
        full_ids = tokenizer(
            f"Question: {q}\nAnswer: {a}",
            return_tensors='pt', add_special_tokens=True,
            truncation=True, max_length=max_length,
        ).input_ids[0]

        q_ids = tokenizer(
            f"Question: {q}\nAnswer: ",
            return_tensors='pt', add_special_tokens=True,
            truncation=True, max_length=max_length,
        ).input_ids[0]

        q_len = q_ids.shape[0]
        if full_ids.shape[0] - q_len <= 0:
            continue

        samples.append((full_ids, q_len))
        valid_pairs.append((q, a))

    return samples, valid_pairs


def compute_batch_loss(
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


def train_decoder_on_model(
    probe_model,
    tokenizer,
    questions:   List[str],
    answers:     List[str],
    split_layer: int,
    max_epochs:  int   = 50,
    batch_size:  int   = 4,
    lr:          float = 5e-5,
    max_length:  int   = 2048,
    val_ratio:   float = 0.2,
    patience:    int   = 3,
    min_epochs:  int   = 3,
    seed:        int   = 42,
) -> Tuple[TrainableDecoder, dict, List[str], List[str]]:

    embed_device = next(probe_model.model.embed_tokens.parameters()).device
    torch.manual_seed(seed)

    all_samples, valid_pairs = build_samples(
        tokenizer, questions, answers, max_length
    )
    N = len(all_samples)

    if N == 0:
        print(f"  ERROR: no valid samples at layer {split_layer}")
        return TrainableDecoder(probe_model), {}, [], []

    n_val   = max(1, int(N * val_ratio))
    n_train = N - n_val
    perm    = torch.randperm(
        N, generator=torch.Generator().manual_seed(seed)
    ).tolist()

    train_samples = [all_samples[i]    for i in perm[:n_train]]
    val_samples   = [all_samples[i]    for i in perm[n_train:]]
    val_questions = [valid_pairs[i][0] for i in perm[n_train:]]
    val_answers   = [valid_pairs[i][1] for i in perm[n_train:]]

    print(f"  Samples: {n_train} train / {n_val} val  "
          f"(layer={split_layer}  max_epochs={max_epochs}  patience={patience})")

    decoder   = TrainableDecoder(probe_model)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)

    test_ids = train_samples[0][0].unsqueeze(0).to(embed_device)
    with torch.no_grad():
        h_test  = get_hidden_state_at_layer(probe_model, test_ids, split_layer)
        lg_test = decoder(h_test)
    print(f"  Diagnostic: hidden_max={h_test.abs().max():.1f}  "
          f"logit_max={lg_test.abs().max():.1f}  "
          f"nan={lg_test.isnan().any().item()}  "
          f"decoder_device={decoder.decoder_device}")

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
        val_loss = compute_batch_loss(
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
    decoder.eval()
    print(f"  Best: epoch={best_epoch}  val_loss={best_val_loss:.4f}  "
          f"val_size={len(val_questions)}")

    info = {
        'best_epoch':    best_epoch,
        'best_val_loss': best_val_loss,
        'history':       history,
        'n_train':       n_train,
        'n_val':         n_val,
    }
    return decoder, info, val_questions, val_answers


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
                'layer':              ell,
                'model':              name,
                'best_epoch':         r['best_epoch'],
                'best_val_loss':      round(r['best_val_loss'], 4),
                'n_val':              r['n_val'],
                'km_original_full':   round(r['km_original_full'], 2),
                'km_own_full':        round(r['km_own_full'], 2),
                'km_unlearned_val':   round(r['km_unlearned_val'], 2),
                'km_dec_val_f':       round(r['km_dec_val_f'], 2),
                'km_dec_val_r':       round(r['km_dec_val_r'], 2),
                'km_direct_val_f':    round(r['km_direct_val_f'], 2),
                'km_direct_val_r':    round(r['km_direct_val_r'], 2),
                'km_direct_full_f':   round(r['km_direct_full_f'], 2),
                'km_direct_full_r':   round(r['km_direct_full_r'], 2),
                'vs_own_f':           round(r['vs_own_f'], 2),
                'vs_original_f':      round(r['vs_original_f'], 2),
            })
    return rows


if __name__ == "__main__":

    ORIGINAL_MODEL_PATH = ...
    TOKENIZER_DIR       = ...
    CORPUS              = ...
    TEMP_DIR            = ...

    UNLEARNED_MODEL_PATHS = {
        "BLUR-NPO": ...,
        "GradDiff": ...,
        "NPO":      ...,
        "SimNPO":   ...,
        "SAM":  ...,
        "PARS": ...,
    }

    MAX_EPOCHS     = 50
    MIN_EPOCHS     = 3
    PATIENCE       = 3
    VAL_RATIO      = 0.2
    LR             = 5e-5
    BATCH_SIZE     = 4
    SEED           = 42
    MAX_NEW_TOKENS = 32

    NUM_LAYERS  = 32
    EVAL_LAYERS = list(range(NUM_LAYERS))

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    qa_f  = read_json(DEFAULT_DATA[CORPUS]['knowmem_forget_qa_file'])
    icl_f = read_json(DEFAULT_DATA[CORPUS]['knowmem_forget_qa_icl_file'])
    questions = [d['question'] for d in qa_f]
    answers   = [d['answer']   for d in qa_f]
    icl_qs    = [d['question'] for d in icl_f]
    icl_as    = [d['answer']   for d in icl_f]

    full_eval_kwargs = dict(
        tokenizer=tokenizer,
        questions=questions,
        answers=answers,
        icl_qs=icl_qs,
        icl_as=icl_as,
        max_new_tokens=MAX_NEW_TOKENS,
    )

    print("Loading original model...")
    original_model = AutoModelForCausalLM.from_pretrained(
        ORIGINAL_MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    )
    original_model.eval()

    print("-" * 70)
    print("Baselines on full forget set")
    print("-" * 70)

    agg, _ = eval_knowmem(model=original_model, **full_eval_kwargs)
    km_original_full = agg['rougeL_f_mean'] * 100   
    write_json(agg, os.path.join(TEMP_DIR, "baselines/original_full.json"))
    print(f"  Original: F={km_original_full:.2f}  "
          f"R={agg['rougeL_r_mean']*100:.2f}")

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

        agg, _ = eval_knowmem(model=unlearned_model, **full_eval_kwargs)
        km_own_full_f = agg['rougeL_f_mean'] * 100
        km_own_full_r = agg['rougeL_r_mean'] * 100
        write_json(agg, os.path.join(TEMP_DIR, f"baselines/{name}_full.json"))
        print(f"  {name} output-level KnowMem: "
              f"F={km_own_full_f:.2f}  R={km_own_full_r:.2f}")

        all_layer_results[name] = {}

        for ell in EVAL_LAYERS:
            print("-" * 70)
            print(f"  [{name}] Layer {ell} / {NUM_LAYERS-1}")
            print("-" * 70)

            decoder, info, val_qs, val_as = train_decoder_on_model(
                probe_model=unlearned_model,
                tokenizer=tokenizer,
                questions=questions,
                answers=answers,
                split_layer=ell,
                max_epochs=MAX_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LR,
                val_ratio=VAL_RATIO,
                patience=PATIENCE,
                min_epochs=MIN_EPOCHS,
                seed=SEED,
            )

            if not info or len(val_qs) == 0:
                print(f"  ERROR: skipping layer {ell}")
                continue

            val_eval_kwargs = dict(
                tokenizer=tokenizer,
                questions=val_qs,
                answers=val_as,
                icl_qs=icl_qs,
                icl_as=icl_as,
                max_new_tokens=MAX_NEW_TOKENS,
            )

            print(f"\n  Baselines on val split ({len(val_qs)} samples):")

            agg, _ = eval_knowmem(model=original_model, **val_eval_kwargs)
            km_original_val_f = agg['rougeL_f_mean'] * 100
            km_original_val_r = agg['rougeL_r_mean'] * 100
            print(f"  Original val: F={km_original_val_f:.2f}  "
                  f"R={km_original_val_r:.2f}")

            agg, _ = eval_knowmem(model=unlearned_model, **val_eval_kwargs)
            km_unlearned_val_f = agg['rougeL_f_mean'] * 100
            km_unlearned_val_r = agg['rougeL_r_mean'] * 100
            print(f"  {name:<20} val: F={km_unlearned_val_f:.2f}  "
                  f"R={km_unlearned_val_r:.2f}")

            decoder.freeze()

            print(f"\n  Frozen decoder on {name} R^({ell}) [val split]")
            hybrid_val = HybridWithTrainedDecoder(
                backbone_model=unlearned_model,
                decoder=decoder,
                split_layer=ell,
            )
            agg_val, _ = eval_knowmem(model=hybrid_val, **val_eval_kwargs)
            km_dec_val_f    = agg_val['rougeL_f_mean'] * 100
            km_dec_val_r    = agg_val['rougeL_r_mean'] * 100
            km_direct_val_f = km_dec_val_f   # alias
            km_direct_val_r = km_dec_val_r   # alias
            print(f"  dec_val: F={km_dec_val_f:.2f}  R={km_dec_val_r:.2f}  "
                  f"(unlearned_val F={km_unlearned_val_f:.2f}  "
                  f"original_val F={km_original_val_f:.2f})")
            write_json(agg_val, os.path.join(
                TEMP_DIR, name, f"layer_{ell}/direct_val_agg.json"
            ))
            del hybrid_val

            print(f"\n  Frozen decoder on {name} R^({ell}) [full set]")
            hybrid_full = HybridWithTrainedDecoder(
                backbone_model=unlearned_model,
                decoder=decoder,
                split_layer=ell,
            )
            agg_full, log_full = eval_knowmem(
                model=hybrid_full, **full_eval_kwargs
            )
            km_direct_full_f = agg_full['rougeL_f_mean'] * 100
            km_direct_full_r = agg_full['rougeL_r_mean'] * 100
            vs_own_f         = km_direct_full_f - km_own_full_f
            vs_original_f    = km_direct_full_f - km_original_full

            print(f"  direct_full: F={km_direct_full_f:.2f}  "
                  f"R={km_direct_full_r:.2f}  "
                  f"vs {name} F: {vs_own_f:+.2f}  "
                  f"vs original F: {vs_original_f:+.2f}")

            write_json(agg_full,  os.path.join(
                TEMP_DIR, name, f"layer_{ell}/direct_agg.json"
            ))
            write_json(log_full,  os.path.join(
                TEMP_DIR, name, f"layer_{ell}/direct_log.json"
            ))
            write_json(info, os.path.join(
                TEMP_DIR, name, f"layer_{ell}/direct_training_info.json"
            ))
            del hybrid_full

            all_layer_results[name][ell] = {
                'best_epoch':       info['best_epoch'],
                'best_val_loss':    info['best_val_loss'],
                'n_val':            len(val_qs),
                'km_original_full': km_original_full,
                'km_own_full':      km_own_full_f,
                'km_unlearned_val': km_unlearned_val_f,
                'km_dec_val_f':     km_dec_val_f,
                'km_dec_val_r':     km_dec_val_r,
                'km_direct_val_f':  km_direct_val_f,
                'km_direct_val_r':  km_direct_val_r,
                'km_direct_full_f': km_direct_full_f,
                'km_direct_full_r': km_direct_full_r,
                'vs_own_f':         vs_own_f,
                'vs_original_f':    vs_original_f,
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
                      os.path.join(TEMP_DIR, "direct_results.csv"))

        del unlearned_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n  Original KnowMem: F={km_original_full:.2f}")

    for name in UNLEARNED_MODEL_PATHS:
        if name not in all_layer_results:
            continue
        print(f"\n  [{name}]")
        for ell in sorted(all_layer_results[name].keys()):
            r = all_layer_results[name][ell]
            print(f"    Layer {ell:>2}  "
                  f"epoch={r['best_epoch']:>2}  "
                  f"dec_val_f={r['km_dec_val_f']:.2f}  "
                  f"dec_val_r={r['km_dec_val_r']:.2f}  "
                  f"direct_full_f={r['km_direct_full_f']:.2f}  "
                  f"direct_full_r={r['km_direct_full_r']:.2f}  "
                  f"vs_own_f={r['vs_own_f']:+.2f}  "
                  f"vs_orig_f={r['vs_original_f']:+.2f}")
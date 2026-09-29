"""
GCG (Greedy Coordinate Gradient) jailbreak attack for unlearning evaluation.

Adapted from: https://github.com/llm-attacks/llm-attacks
Reference: "Universal and Transferable Adversarial Attacks on Aligned Language Models"

For MUSE: goal = first ~20 words of forget passage, target = continuation
For TOFU: goal = question about forgotten author, target = answer

The attack finds an adversarial suffix that maximizes the model's probability
of generating the forgotten target text given the goal prompt.
"""

import argparse
import gc
import json
import os
import re
import string
import csv
import numpy as np
import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge_score import rouge_scorer


def load_model_and_tokenizer(model_path, tokenizer_path=None, device='cuda:0'):
    """Load model and tokenizer, handling different model families."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if str(device).startswith('cpu') else torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        use_cache=False,
    ).to(device).eval()

    tok_path = tokenizer_path if tokenizer_path else model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=True,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    return model, tokenizer


def get_embedding_matrix(model):
    return model.get_input_embeddings().weight


def get_embeddings(model, input_ids):
    return model.get_input_embeddings()(input_ids)


def build_not_allowed_tokens(tokenizer):
    """Exclude special tokens and tokens outside a readable ASCII alphabet."""
    not_allowed = set(tokenizer.all_special_ids)

    allowed_chars = set(
        string.ascii_letters + string.digits +
        " \t"                            # whitespace (no newline)
        ".,!?;:'\"-/()"                  # common punctuation
    )

    for tid in range(len(tokenizer)):
        if tid in not_allowed:
            continue
        decoded = tokenizer.decode([tid])
        if not decoded:
            not_allowed.add(tid)
            continue
        for ch in decoded:
            if ch not in allowed_chars:
                not_allowed.add(tid)
                break
        if tid not in not_allowed and ('<|' in decoded or '|>' in decoded):
            not_allowed.add(tid)

    not_allowed = torch.tensor(sorted(not_allowed), dtype=torch.long)
    print(f"  Built token blacklist: {len(not_allowed)} / {tokenizer.vocab_size} tokens excluded")
    return not_allowed


def token_gradients(model, input_ids, input_slice, target_slice, loss_slice):
    """Compute gradients of the loss w.r.t. each token in input_slice."""
    embed_weights = get_embedding_matrix(model)
    one_hot = torch.zeros(
        input_ids[input_slice].shape[0],
        embed_weights.shape[0],
        device=model.device,
        dtype=embed_weights.dtype,
    )
    one_hot.scatter_(
        1,
        input_ids[input_slice].unsqueeze(1),
        torch.ones(one_hot.shape[0], 1, device=model.device, dtype=embed_weights.dtype),
    )
    one_hot.requires_grad_()
    input_embeds = (one_hot @ embed_weights).unsqueeze(0)

    embeds = get_embeddings(model, input_ids.unsqueeze(0)).detach()
    full_embeds = torch.cat(
        [embeds[:, :input_slice.start, :], input_embeds, embeds[:, input_slice.stop:, :]],
        dim=1,
    )

    logits = model(inputs_embeds=full_embeds).logits
    targets = input_ids[target_slice]
    loss = nn.CrossEntropyLoss()(logits[0, loss_slice, :], targets)
    grad = torch.autograd.grad(loss, one_hot, retain_graph=False)[0].detach()
    grad = grad / grad.norm(dim=-1, keepdim=True).clamp_min(torch.finfo(grad.dtype).eps)
    return grad


def sample_control(control_toks, grad, batch_size, topk=256, not_allowed_tokens=None):
    """Sample a batch of new candidate control tokens based on gradient."""
    if not_allowed_tokens is not None:
        grad[:, not_allowed_tokens.to(grad.device)] = float('inf')

    topk = min(topk, int(torch.isfinite(grad).sum(dim=1).min().item()))
    if topk < 1:
        raise ValueError('No allowed token candidates remain')
    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(batch_size, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / batch_size, device=grad.device
    ).type(torch.int64)
    new_token_val = torch.gather(
        top_indices[new_token_pos], 1,
        torch.randint(0, topk, (batch_size, 1), device=grad.device),
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)
    return new_control_toks


def _build_forbidden_substrings(tokenizer):
    """Exclude special-token strings and repeated control characters."""
    forbidden = set()
    for tok_str in tokenizer.all_special_tokens:
        tok_str = tok_str.strip()
        if tok_str and len(tok_str) >= 3:
            forbidden.add(tok_str)
    forbidden.update([
        '\\\\',            # literal backslash-backslash
        '\n\n',            # actual newline-newline
        '\t\t',            # actual tab-tab
        '}}}}', '{{{{', ']]]]', '[[[[',
        '))))', '((((', '////', '****', '####', '@@@@',
    ])
    return forbidden


def _is_string_clean(decoded_str, tokenizer):
    """Reject special-token text and repeated control characters."""
    if any(pattern in decoded_str for pattern in _build_forbidden_substrings(tokenizer)):
        return False
    if re.search(r'<\|[^>]*\|>', decoded_str):
        return False
    if len(decoded_str) > 0:
        alpha = sum(1 for c in decoded_str if c.isalpha() or c.isspace())
        if alpha / len(decoded_str) < 0.3:
            return False
    return True


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None):
    """Keep candidates with a stable token count; otherwise retain the suffix."""
    cands = []
    for tokens in control_cand:
        text = tokenizer.decode(tokens, skip_special_tokens=False)
        if not filter_cand or (
            text != curr_control
            and len(tokenizer(text, add_special_tokens=False).input_ids) == len(tokens)
            and _is_string_clean(text, tokenizer)
        ):
            cands.append(text)
    if not cands:
        if curr_control is None:
            raise ValueError("No valid candidates and no current suffix")
        cands = [curr_control]
    return cands + [cands[-1]] * (len(control_cand) - len(cands))


def get_logits(model, tokenizer, input_ids, control_slice, test_controls,
               return_ids=False):
    """Evaluate candidates whose token lengths match the control slice."""
    controls = [tokenizer(text, add_special_tokens=False).input_ids for text in test_controls]
    if not controls or any(len(ids) != control_slice.stop - control_slice.start for ids in controls):
        raise ValueError("Candidate suffixes must preserve the control token length")
    ids = input_ids.unsqueeze(0).repeat(len(controls), 1)
    ids[:, control_slice] = torch.tensor(controls, device=ids.device, dtype=ids.dtype)
    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
    return (logits, ids) if return_ids else logits


def target_loss(logits, ids, target_slice):
    """Compute cross-entropy loss on the target slice."""
    crit = nn.CrossEntropyLoss(reduction='none')
    loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)
    loss = crit(logits[:, loss_slice, :].transpose(1, 2), ids[:, target_slice])
    return loss.mean(dim=-1)


class GCGMultiPromptAttack:
    """
    Multi-prompt GCG: optimize one adversarial suffix across multiple goal-target pairs.
    This finds a *universal* suffix that extracts knowledge across many forget samples.
    """

    def __init__(self, model, tokenizer, goals, targets, adv_suffix_init,
                 target_prefix_tokens=16, device='cuda:0'):
        if not goals or len(goals) != len(targets) or target_prefix_tokens < 1:
            raise ValueError('Provide paired goals/targets and a positive prefix length')
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.goals = goals
        self.targets = targets
        self.target_prefix_tokens = target_prefix_tokens
        self.adv_suffix = adv_suffix_init
        self.bos_toks = ([tokenizer.bos_token_id]
                         if tokenizer.bos_token_id is not None else [])

        self.not_allowed_tokens = torch.cat([
            build_not_allowed_tokens(tokenizer),
            torch.arange(len(tokenizer), get_embedding_matrix(model).shape[0]),
        ])

        initial_control_ids = tokenizer(
            adv_suffix_init, add_special_tokens=False
        ).input_ids
        if not initial_control_ids:
            raise ValueError('The initial suffix must contain at least one token')
        self.prompts = []
        for goal, target in zip(goals, targets):
            goal_toks = tokenizer(goal, add_special_tokens=False).input_ids
            full_target_toks = tokenizer(target, add_special_tokens=False).input_ids
            target_toks = full_target_toks[:target_prefix_tokens]
            if not target_toks:
                raise ValueError(f"Target tokenized to an empty sequence: {target!r}")
            target_prefix = tokenizer.decode(target_toks, skip_special_tokens=True)
            toks = self.bos_toks + goal_toks + initial_control_ids + target_toks

            suffix_start = len(self.bos_toks) + len(goal_toks)
            suffix_slice = slice(suffix_start, suffix_start + len(initial_control_ids))
            target_slice = slice(suffix_slice.stop, len(toks))
            loss_slice = slice(target_slice.start - 1, target_slice.stop - 1)

            self.prompts.append({
                'goal': goal,
                'target': target,
                'target_prefix': target_prefix,
                'input_ids': torch.tensor(toks, device=device),
                'control_slice': suffix_slice,
                'target_slice': target_slice,
                'loss_slice': loss_slice,
            })

        self.control_toks = torch.tensor(
            initial_control_ids,
            device=device,
        )
        self.control_slice_len = len(self.control_toks)

    def run(self, n_steps=500, batch_size=512, topk=256, forward_batch_size=16):
        """Run the multi-prompt GCG attack."""
        losses_history = []

        for step in range(n_steps):
            total_grad = None
            for p in self.prompts:
                grad = token_gradients(
                    self.model,
                    p['input_ids'],
                    p['control_slice'],
                    p['target_slice'],
                    p['loss_slice'],
                )
                if total_grad is None:
                    total_grad = grad
                else:
                    total_grad += grad
            total_grad = total_grad / len(self.prompts)

            with torch.no_grad():
                new_control_toks = sample_control(
                    self.control_toks, total_grad, batch_size, topk=topk,
                    not_allowed_tokens=self.not_allowed_tokens,
                )
                new_control_strs = get_filtered_cands(
                    self.tokenizer, new_control_toks,
                    filter_cand=True, curr_control=self.adv_suffix,
                )
                new_control_strs[0] = self.adv_suffix


                all_losses = None
                for p in self.prompts:
                    candidate_losses = []
                    for start in range(0, len(new_control_strs), forward_batch_size):
                        logits, candidate_ids = get_logits(
                            self.model, self.tokenizer, p['input_ids'],
                            p['control_slice'],
                            new_control_strs[start:start + forward_batch_size],
                            return_ids=True,
                        )
                        candidate_losses.append(target_loss(logits, candidate_ids, p['target_slice']))
                        del logits, candidate_ids
                    losses = torch.cat(candidate_losses)
                    if all_losses is None:
                        all_losses = losses
                    else:
                        all_losses += losses
                    del candidate_losses, losses
                    gc.collect()

                all_losses = all_losses / len(self.prompts)
                best_idx = all_losses.argmin()
                best_loss = all_losses[best_idx].item()
                self.adv_suffix = new_control_strs[best_idx]

                self.control_toks = torch.tensor(
                    self.tokenizer(self.adv_suffix, add_special_tokens=False).input_ids,
                    device=self.device,
                )
                if len(self.control_toks) != self.control_slice_len:
                    raise RuntimeError("Selected suffix changed control token length")
                for p in self.prompts:
                    p['input_ids'][p['control_slice']] = self.control_toks

                losses_history.append(best_loss)

            if step % 50 == 0 or step == n_steps - 1:
                print(f"Step {step}/{n_steps} | Loss: {best_loss:.4f}")

            if step > 50 and len(set(losses_history[-20:])) <= 3:
                print(f"  Converged at step {step}")
                break

            del total_grad, new_control_toks
            gc.collect()
            torch.cuda.empty_cache()

        return {
            'adv_suffix': self.adv_suffix,
            'loss_history': losses_history,
            'final_loss': losses_history[-1] if losses_history else None,
        }

    def _build_input_with_suffix(self, prompt_info):
        """Build input_ids with current adversarial suffix."""
        goal_ids = self.tokenizer(
            prompt_info['goal'], add_special_tokens=False, return_tensors='pt'
        ).input_ids[0]
        control_ids = self.tokenizer(
            self.adv_suffix, add_special_tokens=False, return_tensors='pt'
        ).input_ids[0]
        prefix_ids = torch.tensor(self.bos_toks, dtype=goal_ids.dtype)
        return torch.cat([prefix_ids, goal_ids, control_ids])

    def _generate(self, input_ids, max_new_tokens=128):
        """Generate text from input_ids."""
        input_ids = input_ids.unsqueeze(0).to(self.device)
        attn_mask = torch.ones_like(input_ids).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids, attention_mask=attn_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )[0]
        return self.tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True)

    def test_extraction(self, test_goals, test_targets, max_new_tokens=128):
        """Test knowledge extraction on held-out goals/targets using the optimized suffix."""
        results = []
        for goal, target in zip(test_goals, test_targets):
            target_ids = self.tokenizer(
                target, add_special_tokens=False
            ).input_ids[:self.target_prefix_tokens]
            target_eval_ids = self.tokenizer(
                target, add_special_tokens=True
            ).input_ids[:max_new_tokens]
            target_prefix = self.tokenizer.decode(
                target_ids, skip_special_tokens=True
            )
            input_ids = self._build_input_with_suffix({'goal': goal})
            generation = self._generate(input_ids, max_new_tokens=max_new_tokens)
            results.append({
                'goal': goal,
                'target': target,
                'target_eval': self.tokenizer.decode(
                    target_eval_ids, skip_special_tokens=True
                ),
                'target_prefix': target_prefix,
                'generation': generation,
            })
        return results


def compute_extraction_metrics(results):
    """Compute MUSE-compatible VerbMem and target-prefix VerbMem metrics.

    MUSE's official VerbMem metric is mean ROUGE-L F1 (not an exact-match
    rate, and not the hand-written word-LCS recall previously used here).
    Prefix-VerbMem uses the same scorer against the first target-prefix tokens.
    """
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    verbmem_scores = []
    verbmem_recall_scores = []
    prefix_verbmem_scores = []
    prefix_verbmem_recall_scores = []
    exact_matches = []
    for r in results:
        reference = r.get('target_eval', r['target'])
        full = scorer.score(reference, r['generation'])['rougeL']
        prefix = scorer.score(
            r.get('target_prefix', reference), r['generation']
        )['rougeL']
        verbmem_scores.append(full.fmeasure)
        verbmem_recall_scores.append(full.recall)
        prefix_verbmem_scores.append(prefix.fmeasure)
        prefix_verbmem_recall_scores.append(prefix.recall)
        exact_matches.append(1.0 if r['generation'].strip().lower() == reference.strip().lower() else 0.0)

    return {
        'verbmem_mean_rouge_l': np.mean(verbmem_scores) if verbmem_scores else 0.0,
        'verbmem_mean_rouge_l_recall': np.mean(verbmem_recall_scores) if verbmem_recall_scores else 0.0,
        'prefix_verbmem_mean_rouge_l': np.mean(prefix_verbmem_scores) if prefix_verbmem_scores else 0.0,
        'prefix_verbmem_mean_rouge_l_recall': np.mean(prefix_verbmem_recall_scores) if prefix_verbmem_recall_scores else 0.0,
        'exact_match_rate': np.mean(exact_matches) if exact_matches else 0.0,
        'n_samples': len(results),
    }


def generate_baseline_results(attack, tokenizer, goals, targets, max_new_tokens,
                              target_prefix_tokens):
    """Generate the no-suffix MUSE VerbMem baseline for a fixed sample set."""
    results = []
    for goal, target in zip(goals, targets):
        target_prefix_ids = tokenizer(
            target, add_special_tokens=False
        ).input_ids[:target_prefix_tokens]
        target_eval_ids = tokenizer(
            target, add_special_tokens=True
        ).input_ids[:max_new_tokens]
        input_ids = tokenizer(goal, add_special_tokens=True,
                              return_tensors='pt').input_ids[0]
        results.append({
            'goal': goal,
            'target': target,
            'target_eval': tokenizer.decode(target_eval_ids, skip_special_tokens=True),
            'target_prefix': tokenizer.decode(target_prefix_ids, skip_special_tokens=True),
            'generation': attack._generate(input_ids, max_new_tokens=max_new_tokens),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="GCG Jailbreak for Unlearning Evaluation")
    parser.add_argument('--model_path', required=True, help="Path to the unlearned model")
    parser.add_argument('--tokenizer_path', default=None, help="Tokenizer path (defaults to model_path)")
    parser.add_argument('--data_csv', required=True, help="CSV with 'goal' and 'target' columns")
    parser.add_argument('--output_dir', required=True, help="Directory to save results")
    parser.add_argument('--n_train', type=int, default=15, help="Number of training samples for GCG")
    parser.add_argument('--n_test', type=int, default=25, help="Number of test samples for evaluation")
    parser.add_argument('--n_steps', type=int, default=200, help="GCG optimization steps")
    parser.add_argument('--batch_size', type=int, default=128, help="Candidate batch size")
    parser.add_argument('--forward_batch_size', type=int, default=16, help='Candidates per model forward pass')
    parser.add_argument('--topk', type=int, default=256, help="Top-K token candidates")
    parser.add_argument('--adv_suffix_init', default="x x x x x x x x x x",
                        help="Initial adversarial suffix (must round-trip through tokenizer)")
    parser.add_argument('--max_new_tokens', type=int, default=128, help="Max tokens to generate during testing")
    parser.add_argument('--target_prefix_tokens', type=int, default=16,
                        help="Optimize and evaluate this many leading target tokens")
    parser.add_argument('--device', default='cuda:0', help="CUDA device")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    if min(args.n_train, args.n_test, args.batch_size, args.forward_batch_size, args.topk, args.max_new_tokens, args.target_prefix_tokens) < 1 or args.n_steps < 0:
        parser.error('Counts must be positive and n_steps must be nonnegative')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading data from {args.data_csv}")
    all_goals, all_targets = [], []
    with open(args.data_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row['goal'].strip() or not row['target'].strip():
                raise ValueError('CSV goals and targets must be non-empty')
            all_goals.append(row['goal'])
            all_targets.append(row['target'])
    print(f"  Total samples: {len(all_goals)}")

    if len(all_goals) < args.n_train + args.n_test:
        parser.error('The CSV must contain at least n_train + n_test valid rows')
    n_train, n_test = args.n_train, args.n_test
    train_indices = list(range(n_train))
    test_indices = list(range(n_train, n_train + n_test))

    train_goals = [all_goals[i] for i in train_indices]
    train_targets = [all_targets[i] for i in train_indices]
    test_goals = [all_goals[i] for i in test_indices]
    test_targets = [all_targets[i] for i in test_indices]

    print(f"  Train: {len(train_goals)}, Test: {len(test_goals)}")

    print(f"\nLoading model: {args.model_path}")
    model, tokenizer = load_model_and_tokenizer(
        args.model_path, args.tokenizer_path, device=args.device,
    )
    print(f"  Model loaded. Vocab size: {tokenizer.vocab_size}")

    print(f"\nRunning GCG attack ({args.n_steps} steps)...")
    attack = GCGMultiPromptAttack(
        model, tokenizer, train_goals, train_targets,
        adv_suffix_init=args.adv_suffix_init,
        target_prefix_tokens=args.target_prefix_tokens, device=args.device,
    )

    attack_result = attack.run(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        forward_batch_size=args.forward_batch_size,
    )

    print(f"\nOptimized suffix: {repr(attack_result['adv_suffix'])}")
    if attack_result['final_loss'] is not None:
        print(f"Final loss: {attack_result['final_loss']:.4f}")
    else:
        print(f"Final loss: N/A (no successful steps)")

    print(f"\nTesting knowledge extraction on {len(test_goals)} held-out samples...")
    extraction_results = attack.test_extraction(
        test_goals, test_targets, max_new_tokens=args.max_new_tokens,
    )

    train_extraction_results = attack.test_extraction(
        train_goals, train_targets, max_new_tokens=args.max_new_tokens,
    )

    print("Testing baseline (no suffix)...")
    baseline_results = generate_baseline_results(
        attack, tokenizer, test_goals, test_targets,
        args.max_new_tokens, args.target_prefix_tokens,
    )

    print(f"Testing full-dataset baseline on all {len(all_goals)} samples...")
    full_baseline_results = generate_baseline_results(
        attack, tokenizer, all_goals, all_targets,
        args.max_new_tokens, args.target_prefix_tokens,
    )

    gcg_metrics = compute_extraction_metrics(extraction_results)
    train_gcg_metrics = compute_extraction_metrics(train_extraction_results)
    baseline_metrics = compute_extraction_metrics(baseline_results)
    full_baseline_metrics = compute_extraction_metrics(full_baseline_results)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Baseline (no attack): VerbMem={baseline_metrics['verbmem_mean_rouge_l']:.4f}, "
          f"Prefix-VerbMem={baseline_metrics['prefix_verbmem_mean_rouge_l']:.4f}, "
          f"EM={baseline_metrics['exact_match_rate']:.4f}")
    print(f"Full-dataset baseline: VerbMem={full_baseline_metrics['verbmem_mean_rouge_l']:.4f}, "
          f"Prefix-VerbMem={full_baseline_metrics['prefix_verbmem_mean_rouge_l']:.4f}, "
          f"EM={full_baseline_metrics['exact_match_rate']:.4f}")
    print(f"GCG Attack:           VerbMem={gcg_metrics['verbmem_mean_rouge_l']:.4f}, "
          f"Prefix-VerbMem={gcg_metrics['prefix_verbmem_mean_rouge_l']:.4f}, "
          f"EM={gcg_metrics['exact_match_rate']:.4f}")
    print(f"GCG Attack (train):   VerbMem={train_gcg_metrics['verbmem_mean_rouge_l']:.4f}, "
          f"Prefix-VerbMem={train_gcg_metrics['prefix_verbmem_mean_rouge_l']:.4f}, "
          f"EM={train_gcg_metrics['exact_match_rate']:.4f}")

    output = {
        'adv_suffix': attack_result['adv_suffix'],
        'final_loss': attack_result['final_loss'],
        'loss_history': attack_result['loss_history'],
        'baseline_metrics': baseline_metrics,
        'gcg_metrics': gcg_metrics,
        'train_gcg_metrics': train_gcg_metrics,
        'full_dataset_baseline_metrics': full_baseline_metrics,
        'extraction_results': extraction_results,
        'train_extraction_results': train_extraction_results,
        'baseline_results': baseline_results,
        'full_dataset_baseline_results': full_baseline_results,
        'config': {
            'n_train': n_train,
            'n_test': len(test_goals),
            'n_steps': args.n_steps,
            'batch_size': args.batch_size,
            'topk': args.topk,
            'target_prefix_tokens': args.target_prefix_tokens,
        },
    }

    out_path = os.path.join(args.output_dir, 'gcg_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

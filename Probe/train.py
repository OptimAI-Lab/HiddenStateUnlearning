import os
import torch
import torch.nn as nn
import copy
from typing import List, Tuple, Optional
from models import TrainableDecoder, get_hidden_state_at_layer
from utils import write_csv

def build_samples(tokenizer, questions: List[str], answers: List[str], max_length: int = 2048) -> Tuple[List[Tuple[torch.Tensor, int]], List[Tuple[str, str]]]:
    samples = []
    valid_pairs = []
    for q, a in zip(questions, answers):
        full_ids = tokenizer(
            f"Question: {q}\nAnswer: {a}", return_tensors='pt',
            add_special_tokens=True, truncation=True, max_length=max_length
        ).input_ids[0]

        q_ids = tokenizer(
            f"Question: {q}\nAnswer: ", return_tensors='pt',
            add_special_tokens=True, truncation=True, max_length=max_length
        ).input_ids[0]

        q_len = q_ids.shape[0]
        if full_ids.shape[0] - q_len > 0:
            samples.append((full_ids, q_len))
            valid_pairs.append((q, a))
    return samples, valid_pairs

def compute_batch_loss(decoder, target_model, samples, split_layer, batch_size, tokenizer, no_grad=True) -> Optional[float]:
    embed_device = next(target_model.model.embed_tokens.parameters()).device
    total_loss, n_batches = 0.0, 0
    ctx = torch.no_grad() if no_grad else torch.enable_grad()

    with ctx:
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            max_len = max(s[0].shape[0] for s in batch)
            pad_id = tokenizer.pad_token_id

            batch_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

            for b, (ids, q_len) in enumerate(batch):
                seq_len = ids.shape[0]
                batch_ids[b, :seq_len] = ids
                labels[b, q_len:seq_len] = ids[q_len:seq_len]

            batch_ids = batch_ids.to(embed_device)
            labels    = labels.to(decoder.decoder_device)

            if (labels != -100).sum() == 0: continue

            hidden = get_hidden_state_at_layer(target_model, batch_ids, split_layer)
            logits = decoder(hidden)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            if not (torch.isnan(loss) or torch.isinf(loss)):
                total_loss += loss.item()
                n_batches  += 1

    return total_loss / n_batches if n_batches > 0 else None

def train_decoder_early_stopping(
    target_model, tokenizer,
    train_qs: List[str], train_as: List[str],
    val_qs: List[str], val_as: List[str],
    split_layer: int, model_name: str, log_dir: str,
    device_id: int = 0,
    max_epochs: int = 50, batch_size: int = 4, lr: float = 5e-5, max_length: int = 2048,
    patience: int = 3, min_epochs: int = 3, seed: int = 42
):
    embed_device = next(target_model.model.embed_tokens.parameters()).device
    torch.manual_seed(seed)

    os.makedirs(log_dir, exist_ok=True)
    live_log_path = os.path.join(log_dir, f"{model_name.replace('/', '_')}_layer_{split_layer}_training.csv")

    train_samples, _ = build_samples(tokenizer, train_qs, train_as, max_length)
    val_samples, _   = build_samples(tokenizer, val_qs, val_as, max_length)

    n_train = len(train_samples)
    n_val = len(val_samples)

    if n_train == 0 or n_val == 0:
        return TrainableDecoder(target_model, device_id=device_id), {}

    decoder = TrainableDecoder(target_model, device_id=device_id)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state_dict = copy.deepcopy(decoder.state_dict())
    best_epoch = 0
    patience_count = 0

    print(f"    [GPU {device_id}] -> Starting Training: Layer {split_layer} | Train: {n_train} | Val: {n_val}")

    for epoch in range(max_epochs):
        epoch_loss, n_batches = 0.0, 0
        perm_train = torch.randperm(n_train).tolist()

        decoder.train()
        for i in range(0, n_train, batch_size):
            batch = [train_samples[j] for j in perm_train[i:i+batch_size]]
            max_len = max(s[0].shape[0] for s in batch)

            batch_ids = torch.full((len(batch), max_len), tokenizer.pad_token_id, dtype=torch.long)
            labels = torch.full((len(batch), max_len), -100, dtype=torch.long)

            for b, (ids, q_len) in enumerate(batch):
                seq_len = ids.shape[0]
                batch_ids[b, :seq_len] = ids
                labels[b, q_len:seq_len] = ids[q_len:seq_len]

            batch_ids = batch_ids.to(embed_device)
            labels = labels.to(decoder.decoder_device)
            if (labels != -100).sum() == 0: continue

            hidden = get_hidden_state_at_layer(target_model, batch_ids, split_layer)
            logits = decoder(hidden)

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            if torch.isnan(loss) or torch.isinf(loss): continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')

        decoder.eval()
        val_loss = compute_batch_loss(
            decoder, target_model, val_samples, split_layer, batch_size, tokenizer, no_grad=True
        )

        if val_loss is None: continue

        write_csv([{
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4)
        }], live_log_path, append=True)

        # ADDED VISIBILITY: Print every 5 epochs so you know it's working
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"      [GPU {device_id} - Layer {split_layer}] Epoch {epoch+1:>2}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if epoch + 1 >= min_epochs and val_loss < best_val_loss:
            best_val_loss, best_state_dict, best_epoch, patience_count = val_loss, copy.deepcopy(decoder.state_dict()), epoch + 1, 0
        elif epoch + 1 >= min_epochs:
            patience_count += 1

        if epoch + 1 >= min_epochs and patience_count >= patience:
            print(f"      [GPU {device_id} - Layer {split_layer}] Early Stop at Epoch {epoch+1} (Best Val: {best_val_loss:.4f})")
            break

    decoder.load_state_dict(best_state_dict)
    decoder.eval()

    info = {'best_epoch': best_epoch, 'best_val_loss': best_val_loss}
    return decoder, info

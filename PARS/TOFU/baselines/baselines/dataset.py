from .utils import read_text, pad_or_trim_tensor

from typing import List, Tuple
from pathlib import Path
import json

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


class DefaultDataset(Dataset):

    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        # Keep tokenizer reference for collate functions
        self.tokenizer = tokenizer
        # Support both JSON arrays and JSONL (one JSON object per line).
        p = Path(file_path)
        if p.suffix.lower() in ('.json', '.jsonl'):
            print("path,",file_path)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            # Try full JSON first (a list); otherwise fall back to JSONL
            try:
                data = json.loads(text)
            except Exception:
                data = []
                for line in text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

            if len(data) == 0:
                raise ValueError(f"No entries found in {file_path}")

            first = data[0]

            # Simple list of strings
            if isinstance(first, str):
                self.strings = data

                assert tokenizer is not None, "Tokenizer must be specified."

                self.input_ids = []
                for s in self.strings:
                    encoding: torch.Tensor = tokenizer(
                        s,
                        add_special_tokens=add_bos_token,
                        return_tensors='pt'
                    ).input_ids[0]
                    encoding = pad_or_trim_tensor(
                        encoding,
                        target_length=max_len,
                        padding_value=tokenizer.pad_token_id
                    )
                    self.input_ids.append(encoding)

                return

            # Generic list of dicts with `text` field
            elif isinstance(first, dict) and 'text' in first and isinstance(first['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in first:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return

                assert tokenizer is not None, "Tokenizer must be specified."

                self.input_ids = []
                for s in self.strings:
                    encoding: torch.Tensor = tokenizer(
                        s,
                        add_special_tokens=add_bos_token,
                        return_tensors='pt'
                    ).input_ids[0]
                    encoding = pad_or_trim_tensor(
                        encoding,
                        target_length=max_len,
                        padding_value=tokenizer.pad_token_id
                    )
                    self.input_ids.append(encoding)

                return

            # QA-style JSON lines with `question` and `answer` fields
            elif isinstance(first, dict) and 'question' in first and 'answer' in first:
                assert tokenizer is not None, "Tokenizer must be specified."

                self.input_ids = []
                self.labels = []
                for d in data:
                    q = d.get('question', '')
                    a = d.get('answer', '')

                    # Build prompt so that model is conditioned on question and trained
                    # to predict the answer. Labels are -100 for question tokens.
                    question_prefix = f"Question: {q}\nAnswer: "

                    q_enc = tokenizer(
                        question_prefix,
                        add_special_tokens=add_bos_token,
                        return_tensors='pt'
                    ).input_ids[0]

                    full_enc = tokenizer(
                        question_prefix + a,
                        add_special_tokens=add_bos_token,
                        return_tensors='pt'
                    ).input_ids[0]

                    full_enc = pad_or_trim_tensor(
                        full_enc,
                        target_length=max_len,
                        padding_value=tokenizer.pad_token_id
                    )

                    lbl = full_enc.clone()
                    q_len = min(q_enc.size(0), lbl.size(0))
                    lbl[:q_len] = -100
                    lbl[full_enc == tokenizer.pad_token_id] = -100

                    self.input_ids.append(full_enc)
                    self.labels.append(lbl)

                return

            else:
                raise ValueError(f"Format of {file_path} not recognized.")

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)

        pass    # def __init__()


    def __getitem__(self, index):
        # If labels were prepared (QA-mode), return tuple (input_ids, labels)
        if hasattr(self, 'labels'):
            return self.input_ids[index], self.labels[index]
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor | Tuple[torch.Tensor, torch.Tensor]]):
            # Support items that are either tensors or (input_ids, labels) tuples.
            if isinstance(batch[0], tuple):
                input_ids = torch.stack([item[0] for item in batch])
                labels = torch.stack([item[1] for item in batch])
            else:
                input_ids = torch.stack(batch)
                labels = input_ids.clone()

            attention_mask = (input_ids != getattr(self.tokenizer, 'pad_token_id', 0))

            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }

        return collate_fn



class ForgetRetainDataset(DefaultDataset):

    def __init__(
        self,
        forget_file_path: str,
        tokenizer: AutoTokenizer,
        retain_file_path: str | None = None,
        max_len: int = 4096,
        add_bos_token: bool = True
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        if self.retain_exists:
            return (
                self.forget_dataset[index],
                self.retain_dataset[index % len(self.retain_dataset)]
            )
        else:
            return self.forget_dataset[index], None


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
            # Each element of `batch` is either (input_tensor, input_tensor) or
            # ((input_tensor, label_tensor), (input_tensor, label_tensor)) depending
            # on whether the underlying DefaultDataset prepared labels (QA-mode).

            forget_inputs = []
            forget_labels = []
            retain_inputs = []
            retain_labels = []

            for pair in batch:
                f_item, r_item = pair

                if isinstance(f_item, tuple):
                    f_in, f_lbl = f_item
                else:
                    f_in, f_lbl = f_item, f_item.clone()

                forget_inputs.append(f_in)
                forget_labels.append(f_lbl)

                if self.retain_exists:
                    if isinstance(r_item, tuple):
                        r_in, r_lbl = r_item
                    else:
                        r_in, r_lbl = r_item, r_item.clone()
                    retain_inputs.append(r_in)
                    retain_labels.append(r_lbl)

            batch_forget = torch.stack(forget_inputs)
            batch_forget_labels = torch.stack(forget_labels)
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget_labels,
                "attention_mask": (batch_forget != getattr(self.tokenizer, 'pad_token_id', 0))
            }

            if self.retain_exists:
                batch_retain = torch.stack(retain_inputs)
                batch_retain_labels = torch.stack(retain_labels)
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain_labels,
                    "attention_mask": (batch_retain != getattr(self.tokenizer, 'pad_token_id', 0))
                }
            else:
                dict_retain = None

            return dict_forget, dict_retain

        return collate_fn

"""Forget-set datasets for TOFU answer tuning and MUSE language modeling."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class RelearnDataset(Dataset):
    def __init__(self, data_file, tokenizer, benchmark, max_len=4096):
        if benchmark not in {"TOFU", "MUSE"}:
            raise ValueError("benchmark must be TOFU or MUSE")
        if max_len < 2:
            raise ValueError("max_len must be at least 2")
        self.tokenizer = tokenizer
        self.benchmark = benchmark
        self.max_len = max_len
        path = Path(data_file)
        if path.suffix == ".txt" and benchmark == "MUSE":
            self.samples = [path.read_text(encoding="utf-8")]
        elif path.suffix == ".json":
            self.samples = json.loads(path.read_text())
        else:
            raise ValueError("Use a JSON list, or a MUSE plain-text file")
        if not isinstance(self.samples, list) or not self.samples:
            raise ValueError("The forget set must be a non-empty list")
        if benchmark == "MUSE":
            chunks = []
            bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
            for sample in self.samples:
                if isinstance(sample, dict) and "input_ids" in sample:
                    tokens = list(sample["input_ids"])
                    prefix = []
                else:
                    text = sample.get("text") if isinstance(sample, dict) else sample
                    if not isinstance(text, str):
                        raise ValueError("MUSE requires text strings or text/input_ids records")
                    if not text.strip():
                        continue
                    tokens = tokenizer(text, add_special_tokens=False, verbose=False).input_ids
                    prefix = bos
                stride = max_len - len(prefix)
                for start in range(0, len(tokens), stride):
                    ids = prefix + tokens[start:start + stride]
                    if len(ids) == 1 and start > 0:
                        ids = [tokens[start - 1]] + ids
                    if len(ids) >= 2:
                        chunks.append({"input_ids": ids})
            if not chunks:
                raise ValueError("The MUSE forget set contains no next-token training targets")
            self.samples = chunks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt_len = 0
        if self.benchmark == "TOFU":
            if not isinstance(sample, dict):
                raise ValueError("TOFU requires question/answer records")
            question = sample.get("question", sample.get("Question"))
            answer = sample.get("answer", sample.get("Answer"))
            if not isinstance(question, str) or not isinstance(answer, str) or not answer.strip():
                raise ValueError("TOFU requires a question and a non-empty answer")
            prompt = f"Question: {question}\nAnswer:"
            text = prompt + " " + answer
            prompt_len = len(self.tokenizer(prompt).input_ids)
        elif isinstance(sample, str):
            text = sample
        elif isinstance(sample, dict):
            text = sample.get("text")
        else:
            raise ValueError("MUSE requires text strings or text/input_ids records")

        if self.benchmark == "MUSE" and isinstance(sample, dict) and "input_ids" in sample:
            ids = list(sample["input_ids"][:self.max_len])
        else:
            if not isinstance(text, str) or not text.strip():
                raise ValueError("Training text must be non-empty")
            ids = self.tokenizer(text, truncation=True, max_length=self.max_len).input_ids
        if len(ids) < 2 or prompt_len >= len(ids):
            raise ValueError("No training targets remain; check data or increase max_len")
        length = len(ids)
        input_ids = torch.tensor(ids + [self.tokenizer.pad_token_id] * (self.max_len - length))
        attention_mask = torch.arange(self.max_len) < length
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[~attention_mask] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask.long(), "labels": labels}

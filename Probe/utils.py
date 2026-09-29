import os
import json
import csv
import torch
from typing import List
from transformers import pipeline

def read_data(path):
    """Read the prepared TOFU JSON list or an existing JSONL file."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(data, list):
        raise ValueError("Expected a list of TOFU question/answer records")
    return data

def write_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)

def write_csv(rows: List[dict], path: str, append=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        return
    mode = 'a' if append else 'w'
    with open(path, mode, newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if not append or os.path.getsize(path) == 0:
            writer.writeheader()
        writer.writerows(rows)

def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words:
        string = string.split(word)[0]
    return string

class EntailmentEvalLogger:
    def __init__(self, device_id=0):
        """
        Loads a standard NLI model to evaluate entailment.
        device_id: 0 for cuda:0, -1 for CPU.
        """
        print("    [+] Loading DeBERTa MNLI for Entailment Evaluation...")
        self.classifier = pipeline(
            "text-classification",
            model="sileod/deberta-v3-base-tasksource-nli",
            device=device_id
        )
        self.scores = []
        self.logs   = []

    def log(self, prompt, answer, output, question=None):
        # We check if the Generated Output (Premise) entails the True Answer (Hypothesis)
        if output.strip() and answer.strip():
            result = self.classifier(
                [{"text": output, "text_pair": answer}],
                truncation=True, max_length=512,
            )[0]
            is_entailment = 1.0 if result['label'].upper() == 'ENTAILMENT' else 0.0
        else:
            is_entailment = 0.0

        self.scores.append(is_entailment)
        self.logs.append({
            "question": question,
            "prompt":   prompt,
            "answer":   answer,
            "output":   output,
            "entailment_score": is_entailment,
        })

    def report(self):
        mean = sum(self.scores) / len(self.scores) if self.scores else 0.0
        return {"entailment_mean": mean, "entailment_all": self.scores, "logs": self.logs}

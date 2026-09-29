"""Compare TOFU generations before and after relearning using NLI entailment.

Supply local question/answer JSON. A public NLI model is used by default.
For MUSE's benchmark metrics, use the existing PARS/MUSE/eval.py evaluator.
"""

import argparse
import gc
import json
from pathlib import Path
from statistics import mean

DEFAULT_NLI_MODEL = "sileod/deberta-v3-base-tasksource-nli"


def generate_for_model(model_path, data, tokenizer_path=None, max_new_tokens=256,
                       batch_size=16, device="cuda"):
    import torch
    from rouge_score import rouge_scorer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer needs a padding or end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if str(device).startswith("cuda") and torch.cuda.is_bf16_supported() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device).eval()
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    results = []
    for start in range(0, len(data), batch_size):
        batch = data[start:start + batch_size]
        questions = [item.get("question", item.get("Question")) for item in batch]
        answers = [item.get("answer", item.get("Answer")) for item in batch]
        if any(not isinstance(value, str) for value in questions + answers):
            raise ValueError("TOFU evaluation requires question/answer strings")
        prompts = [f"Question: {question}\nAnswer:" for question in questions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generations = tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
        )
        for question, answer, generation in zip(questions, answers, generations):
            generation = generation.strip()
            results.append({
                "question": question, "ground_truth": answer,
                "responses": [{
                    "response": generation,
                    "rougeL_recall": scorer.score(answer, generation)["rougeL"].recall,
                }],
            })
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def evaluate_generated(records, nli_pipeline, entailment_label="entailment"):
    """Score entailment using the original ROUGE-L recall filter."""
    for record in records:
        target = record["ground_truth"]
        for response in record["responses"]:
            generation = response["response"]
            es = 0.0
            if generation and target and response["rougeL_recall"] >= 0.1:
                prediction = nli_pipeline(
                    [{"text": generation, "text_pair": target}],
                    truncation=True, max_length=512,
                )[0]
                es = float(prediction["label"].lower() == entailment_label.lower())
            response["ES"] = es
        values = [response["ES"] for response in record["responses"]]
        record["Best ES"] = max(values) if values else 0.0
        record["Avg ES"] = mean(values) if values else 0.0
    return {
        "Avg_ES_Forget": mean(record["Avg ES"] for record in records),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orig_model", required=True)
    parser.add_argument("--relearn_model_dir", required=True)
    parser.add_argument("--tokenizer_dir", default=None)
    parser.add_argument("--data_file", required=True, help="TOFU forget QA JSON")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--nli_model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--entailment_label", default="entailment")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default=None, help="Defaults to cuda when available, otherwise cpu")
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_new_tokens < 1:
        parser.error("batch_size and max_new_tokens must be positive")
    import torch

    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    data = json.loads(Path(args.data_file).read_text())
    if not isinstance(data, list) or not data:
        parser.error("data_file must contain a non-empty JSON list")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = (("original", args.orig_model), ("relearn", args.relearn_model_dir))
    generated = {}
    for variant, model_path in variants:
        records = generate_for_model(
            model_path, data, args.tokenizer_dir,
            args.max_new_tokens, args.batch_size, args.device,
        )
        generated[variant] = records
        with (out_dir / f"{variant}.jsonl").open("w") as output:
            for record in records:
                output.write(json.dumps(record) + "\n")

    from transformers import pipeline

    nli_pipeline = pipeline("text-classification", model=args.nli_model, device=args.device)
    labels = nli_pipeline.model.config.id2label.values()
    if args.entailment_label.lower() not in {label.lower() for label in labels}:
        raise ValueError("Set entailment_label to the classifier's entailment label")
    summary = {}
    for variant, records in generated.items():
        summary[variant] = evaluate_generated(
            records, nli_pipeline, args.entailment_label,
        )
        with (out_dir / f"{variant}_evaluated.jsonl").open("w") as output:
            for record in records:
                output.write(json.dumps(record) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()

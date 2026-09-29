"""Prepare matching forget/retain sets and benchmark evaluation data for PARS."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Attacks.Relearn.prepare_data import prepare_data, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["TOFU", "MUSE"], required=True)
    parser.add_argument("--forget_split", choices=["forget01", "forget05", "forget10"], default="forget10")
    parser.add_argument("--corpus", choices=["books", "news"], default="news")
    parser.add_argument("--data_dir", help="Override PARS/<benchmark>/data")
    args = parser.parse_args()

    from datasets import load_dataset

    forget_path = prepare_data(**vars(args))
    if args.benchmark == "TOFU":
        retain_split = {"forget01": "retain99", "forget05": "retain95", "forget10": "retain90"}[args.forget_split]
        data = load_dataset("locuslab/TOFU", retain_split, split="train")
        write_json(forget_path.parent / f"{retain_split}.json", [
            {"question": row["question"], "answer": row["answer"]} for row in data
        ])
    else:
        data = load_dataset(f"muse-bench/MUSE-{args.corpus.title()}", "raw", split="retain1")
        passages = list(data["text"])
        write_json(forget_path.parent / "retain1.json", passages)
        retain_path = forget_path.parent / "retain1.txt"
        retain_path.write_text("\n\n".join(passages), encoding="utf-8")
        print(f"Saved retain corpus to {retain_path}")


if __name__ == "__main__":
    main()

"""Download the forget set and evaluation data for TOFU or MUSE."""

import argparse
import json
from pathlib import Path
import warnings


def write_json(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} records to {path}")


def prepare_data(benchmark, data_dir=None, forget_split="forget10", corpus="books"):
    from datasets import load_dataset
    from datasets.utils.info_utils import NonMatchingSplitsSizesError

    if benchmark not in {"TOFU", "MUSE"}:
        raise ValueError("benchmark must be TOFU or MUSE")
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "PARS" / benchmark / "data"
    if benchmark == "TOFU":
        if forget_split not in {"forget01", "forget05", "forget10"}:
            raise ValueError("Select forget01, forget05, or forget10")
        try:
            data = load_dataset("locuslab/TOFU", forget_split, split="train")
        except NonMatchingSplitsSizesError:
            warnings.warn("TOFU split-size metadata is stale; exporting all actual QA rows.")
            data = load_dataset("locuslab/TOFU", forget_split, split="train", verification_mode="no_checks")
        if any(not isinstance(row.get(key), str) or not row[key].strip()
               for row in data for key in ("question", "answer")):
            raise ValueError("TOFU records must contain non-empty question/answer strings")
        path = root / "tofu" / f"{forget_split}.json"
        write_json(path, [{"question": row["question"], "answer": row["answer"]} for row in data])
        return path

    if corpus not in {"books", "news"}:
        raise ValueError("corpus must be books or news")
    repo = f"muse-bench/MUSE-{corpus.title()}"
    directory = root / corpus
    raw = load_dataset(repo, "raw", split="forget")
    passages = list(raw["text"])
    write_json(directory / "raw" / "forget.json", passages)
    path = directory / "raw" / "forget.txt"
    path.write_text("\n\n".join(passages), encoding="utf-8")
    print(f"Saved raw forget corpus to {path}")
    for config, splits in {
        "verbmem": ["forget"],
        "knowmem": ["forget_qa", "forget_qa_icl", "retain_qa", "retain_qa_icl"],
        "privleak": ["forget", "retain", "holdout"],
    }.items():
        for split in splits:
            data = load_dataset(repo, config, split=split)
            records = list(data["text"]) if config == "privleak" else list(data)
            write_json(directory / config / f"{split}.json", records)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["TOFU", "MUSE"], required=True)
    parser.add_argument("--forget_split", choices=["forget01", "forget05", "forget10"], default="forget10")
    parser.add_argument("--corpus", choices=["books", "news"], default="books")
    parser.add_argument("--data_dir", help="Override PARS/<benchmark>/data")
    prepare_data(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()

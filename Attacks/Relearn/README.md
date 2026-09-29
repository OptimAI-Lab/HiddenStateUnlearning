# Relearning

Tests unlearning robustness by measuring whether training on forgotten data restores that information; more recovery at the same training budget means weaker robustness.

## Data preparation

Run from the repository root; TOFU is saved to `PARS/TOFU/data/tofu/forget10.json`, and MUSE training and evaluation data to `PARS/MUSE/data/books/`.

```bash
python Attacks/Relearn/prepare_data.py --benchmark TOFU --forget_split forget10
python Attacks/Relearn/prepare_data.py --benchmark MUSE --corpus books
```

## Run TOFU

Replace the checkpoint and tokenizer paths; evaluation uses NLI (`sileod/deberta-v3-base-tasksource-nli`) and saves `summary.json`.

```bash
python Attacks/Relearn/relearn.py --benchmark TOFU \
  --model_dir /path/to/tofu_checkpoint --tokenizer_dir /path/to/tofu_tokenizer \
  --data_file PARS/TOFU/data/tofu/forget10.json --out_dir outputs/tofu_relearn
python Attacks/Relearn/eval.py --orig_model /path/to/tofu_checkpoint \
  --relearn_model_dir outputs/tofu_relearn --tokenizer_dir /path/to/tofu_tokenizer \
  --data_file PARS/TOFU/data/tofu/forget10.json --out_dir outputs/tofu_relearn_eval
```

## Run MUSE

Train from the repository root, then evaluate from `PARS/MUSE` using its benchmark metrics; for News, replace `books` with `news` throughout.

```bash
python Attacks/Relearn/relearn.py --benchmark MUSE \
  --model_dir /path/to/muse_checkpoint --tokenizer_dir /path/to/muse_tokenizer \
  --data_file PARS/MUSE/data/books/raw/forget.txt --out_dir outputs/muse_relearn

cd PARS/MUSE
python eval.py --model_dirs /path/to/muse_checkpoint ../../outputs/muse_relearn \
  --names original relearn --tokenizer_dir /path/to/muse_tokenizer \
  --corpus books --out_file ../../outputs/muse_relearn_eval.csv
```

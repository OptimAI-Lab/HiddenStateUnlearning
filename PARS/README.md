# PARS unlearning

PARS trains an unlearned model to suppress information recoverable by hidden-state probes, with separate implementations for TOFU and MUSE.

## Data preparation

Run from the repository root; these commands download TOFU's `forget10`/`retain90` sets to `PARS/TOFU/data/tofu/` and MUSE-News training and evaluation data to `PARS/MUSE/data/news/`.

```bash
python PARS/prepare_data.py --benchmark TOFU
python PARS/prepare_data.py --benchmark MUSE --corpus news
```

## Run

Replace the checkpoint and tokenizer paths below; `minimax_npo_gdr` runs PARS with NPO and retain-set training. Choose probe layers that exist in your model; the current TOFU implementation places probes on GPU 1, so expose at least two GPUs.

```bash
# TOFU
python PARS/TOFU/baselines/unlearn.py --algo minimax_npo_gdr \
  --model_dir /path/to/checkpoint --tokenizer_dir /path/to/tokenizer \
  --data_file PARS/TOFU/data/tofu/forget10.json \
  --retain_data_file PARS/TOFU/data/tofu/retain90.json \
  --probe_layers 8 10 12 14 --out_dir outputs/pars_tofu

# MUSE
python PARS/MUSE/baselines/unlearn.py --algo minimax_npo_gdr \
  --model_dir /path/to/checkpoint --tokenizer_dir /path/to/tokenizer \
  --data_file PARS/MUSE/data/news/raw/forget.txt \
  --retain_data_file PARS/MUSE/data/news/raw/retain1.txt \
  --probe_layers 6 20 25 28 --probe_device 0 --out_dir outputs/pars_muse
```

Use `--help` for training options. Saved checkpoints can be evaluated with [Probe](../Probe/README.md) and the [robustness attacks](../README.md#how-to-use).

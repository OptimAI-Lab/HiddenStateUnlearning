# Jailbreak

Jailbreak tests unlearning robustness by measuring whether an adversarial suffix recovers forgotten information; more recovery means weaker robustness.

## Data preparation

After [downloading the datasets](../Relearn/README.md#data-preparation), run these commands from the repository root to convert TOFU questions/answers and MUSE prompts/continuations into the GCG input files. The attack commands below read these exact files through `--data_csv`; they do not read the original JSON directly.

```bash
python Attacks/Jailbreak/prepare_data.py --benchmark TOFU \
  --forget_data PARS/TOFU/data/tofu/forget10.json --output_csv outputs/tofu.csv
python Attacks/Jailbreak/prepare_data.py --benchmark MUSE \
  --forget_data PARS/MUSE/data/books/verbmem/forget.json --output_csv outputs/muse.csv
```

## Run

Replace the checkpoint and tokenizer paths; results are saved in each output directory as `gcg_results.json`.

```bash
# TOFU
python Attacks/Jailbreak/gcg_attack.py --model_path /path/to/tofu_checkpoint \
  --tokenizer_path /path/to/tofu_tokenizer \
  --data_csv outputs/tofu.csv --output_dir outputs/tofu_jailbreak

# MUSE
python Attacks/Jailbreak/gcg_attack.py --model_path /path/to/muse_checkpoint \
  --tokenizer_path /path/to/muse_tokenizer \
  --data_csv outputs/muse.csv --output_dir outputs/muse_jailbreak
```

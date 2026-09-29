# Hidden-state leakage

Generative probes measure how much forgotten information can still be recovered from a model's hidden states.

## Data preparation

From the repository root, download TOFU to `PARS/TOFU/data/tofu/` and MUSE-News to `PARS/MUSE/data/news/`:

```bash
python PARS/prepare_data.py --benchmark TOFU
python PARS/prepare_data.py --benchmark MUSE --corpus news
```

WMDP-Bio is downloaded automatically from `cais/wmdp` when its probe runs.

## Run

In the TOFU and MUSE scripts, replace the `...` settings with your model, tokenizer, and output paths; keep only the models you want to evaluate. Set TOFU's `DATA_PATH` to `PARS/TOFU/data/tofu/forget10.json`, MUSE's `CORPUS` to `"news"`, and each script's layer settings to match your model.

```bash
# From the repository root
python Probe/probe_tofu.py
python Probe/probe_wmdp.py --model_name target --model_path /path/to/checkpoint

# MUSE probes use paths relative to PARS/MUSE
cd PARS/MUSE
python ../../Probe/probe_muse_verbmem.py
python ../../Probe/probe_muse_knowmem.py
```

The WMDP checkpoint should include its tokenizer; otherwise set `DEFAULT_TOKENIZER_PATH` in the script.

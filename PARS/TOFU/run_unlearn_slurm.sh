#!/bin/bash


# Optional module loads (uncomment & adjust to your cluster)
# module load gcc/11.3.0
# module load anaconda
# module load cuda/12.0
# module list

# Activate conda environment (uncomment as needed)
# conda activate muse_bench
# or
# source activate muse_bench

# --------------------
# Config (customize before submitting)
# --------------------
ALGO=${ALGO:-minimax_ga}
MODEL_DIR=${MODEL_DIR:-open-unlearning/tofu_Llama-3.2-1B-Instruct_full}
DATA_FILE=${DATA_FILE:-data/forget.json}
RETAIN_FILE=${RETAIN_FILE:-data/retain.json}
OUT_DIR=${OUT_DIR:-result/tofu_unlearn_$SLURM_JOB_ID}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
EPOCHS=${EPOCHS:-5}
LR=${LR:-1e-5}
MAX_LEN=${MAX_LEN:-4096}
PROBE_LAYERS=${PROBE_LAYERS:-"8 10 12 14"}
PROBE_LR=${PROBE_LR:-1e-4}
PROBE_INNER_STEPS=${PROBE_INNER_STEPS:-3}
PROBE_BETA=${PROBE_BETA:-0.5}
TOKENIZER_DIR=${TOKENIZER_DIR:-}

mkdir -p log

# Benchmark info
echo "GPU availability:"
nvidia-smi
echo ""
echo "Python executable:"
which python3 || which python
echo ""
echo "TIMING - Starting unlearning at: $(date)"
echo "Job is starting on $(hostname)"
echo ""

# If pulling from private HF repo, export HF_TOKEN or login beforehand
# export HF_TOKEN=your_hf_token
# huggingface-cli login --token $HF_TOKEN

# Build optional tokenizer arg
if [ -n "$TOKENIZER_DIR" ]; then
  TOKENIZER_ARG=(--tokenizer_dir "$TOKENIZER_DIR")
else
  TOKENIZER_ARG=()
fi

# Run training (Trainer/transformers will auto-detect GPUs in many setups)
python3 baselines/unlearn.py \
  --algo "$ALGO" \
  --model_dir "$MODEL_DIR" \
  --data_file "$DATA_FILE" \
  --retain_data_file "$RETAIN_FILE" \
  --out_dir "$OUT_DIR" \
  --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --lr "$LR" \
  --max_len "$MAX_LEN" \
  --probe_layers $PROBE_LAYERS \
  --probe_lr "$PROBE_LR" \
  --probe_inner_steps "$PROBE_INNER_STEPS" \
  --probe_beta "$PROBE_BETA" "${TOKENIZER_ARG[@]}"

EXIT_CODE=$?
echo "TIMING - Finished at: $(date) (exit code: $EXIT_CODE)"
exit $EXIT_CODE

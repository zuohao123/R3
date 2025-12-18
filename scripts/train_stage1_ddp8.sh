#!/usr/bin/env bash
set -euo pipefail

# Stage1: 干净训练（无 PMC）——对齐基座能力，主要训练 LoRA（R³路径关闭）。
#
# 用法：
#   bash scripts/train_stage1_ddp8.sh [OUTDIR]
#
# 监控：
#   tail -f <OUTDIR>/console.log
#   tail -f <OUTDIR>/train.log

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_ASYNC_ERROR_HANDLING=1

if [[ -n "${NPROC_PER_NODE:-}" ]]; then
  NPROC_PER_NODE="${NPROC_PER_NODE}"
else
  IFS=',' read -ra _GPU_ARR <<< "${CUDA_VISIBLE_DEVICES}"
  NPROC_PER_NODE="${#_GPU_ARR[@]}"
fi
OUTDIR="${1:-checkpoints/stage1_clean_ddp8}"
MAX_STEPS="${MAX_STEPS:-3000}"
QUICK_EVAL_EVERY="${QUICK_EVAL_EVERY:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

mkdir -p "${OUTDIR}"

nohup torchrun --nproc_per_node="${NPROC_PER_NODE}" train_r3.py \
  --config configs/default.yaml \
  --device cuda \
  --output_dir "${OUTDIR}" \
  --log_file "${OUTDIR}/train.log" \
  --max_steps "${MAX_STEPS}" \
  --quick_eval_every "${QUICK_EVAL_EVERY}" \
  --log_interval "${LOG_INTERVAL}" \
  > "${OUTDIR}/console.log" 2>&1 &

echo "[OK] Stage1 started: ${OUTDIR}"
echo "  tail -f ${OUTDIR}/console.log"

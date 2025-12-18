#!/usr/bin/env bash
set -euo pipefail

# Stage2: 轻度 PMC（corruption + retrieval + consistency）
#
# 用法：
#   bash scripts/train_stage2_ddp8.sh <STAGE1_CKPT_DIR> [OUTDIR]
#
# 例子：
#   bash scripts/train_stage2_ddp8.sh checkpoints/stage1_clean_ddp8/checkpoint-3000
#
# 监控：
#   tail -f <OUTDIR>/console.log
#   tail -f <OUTDIR>/train.log

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <STAGE1_CKPT_DIR> [OUTDIR]"
  exit 1
fi

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
INIT_CKPT="${1}"
OUTDIR="${2:-checkpoints/stage2_light_pmc_ddp8}"
MAX_STEPS="${MAX_STEPS:-2000}"
QUICK_EVAL_EVERY="${QUICK_EVAL_EVERY:-500}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"

mkdir -p "${OUTDIR}"

nohup torchrun --nproc_per_node="${NPROC_PER_NODE}" train_r3.py \
  --config configs/stage2.yaml \
  --device cuda \
  --output_dir "${OUTDIR}" \
  --log_file "${OUTDIR}/train.log" \
  --init_from_checkpoint "${INIT_CKPT}" \
  --max_steps "${MAX_STEPS}" \
  --quick_eval_every "${QUICK_EVAL_EVERY}" \
  --log_interval "${LOG_INTERVAL}" \
  > "${OUTDIR}/console.log" 2>&1 &

echo "[OK] Stage2 started: ${OUTDIR}"
echo "  init_from_checkpoint=${INIT_CKPT}"
echo "  tail -f ${OUTDIR}/console.log"

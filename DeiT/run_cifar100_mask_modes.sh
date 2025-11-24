#!/bin/bash
# 并行启动多个 mask_mode 的测试/训练（自动填充 job_dir / wandb 名称）
# 与 CnnModels/run.sh 保持一致的监控行为：switch 文件、实时监控、失败检测、可选择启用层
set -e

# ---------- 可改的默认参数 ----------
MODEL="vit_deit_small_patch16_224"
DATA_PATH="/home/yzy/GitHub/Bi-Mask/datasets"
OUTPUT_BASE="experiments/cifar100_deit"
WANDB_PROJECT="bimask_deit_cifar100"
NM_LAYERS="all"
BATCH_SIZE=128
INPUT_SIZE=32
EPOCHS=100
LR=0.001

# Switch File Configuration
SWITCH_FILE="run_switch.txt"

# 要运行的 mask 模式
MODES=(m1 m2 m3 m4 m5)

# 解析 GPU 列表
if [ -n "${1:-}" ]; then
  IFS=',' read -ra GPU_IDS <<< "$1"
else
  if command -v nvidia-smi >/dev/null 2>&1; then
    mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo "0")
  else
    GPU_IDS=(0)
  fi
fi

GPU_COUNT=${#GPU_IDS[@]}
if [ "$GPU_COUNT" -eq 0 ]; then
  GPU_IDS=(0)
  GPU_COUNT=1
fi

echo "Detected GPUs: ${GPU_IDS[*]}"
echo "Modes to run: ${MODES[*]}"

# 自动创建 Switch File (如果不存在)
if [ ! -f "$SWITCH_FILE" ]; then
  echo "ON" > "$SWITCH_FILE"
  echo "Created switch file: $SWITCH_FILE (Content: ON)"
fi

# 捕捉 Ctrl-C / 终止信号
function on_exit {
  echo ""
  echo "Caught signal, terminating all jobs..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "  Killing PID $pid"
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  echo "All child processes terminated."
  exit 1
}
trap on_exit SIGINT SIGTERM

# 主循环：不断开始新的训练 Set (直到 Switch File 变空/消失)
ROUND=0
while true; do
  ROUND=$((ROUND + 1))

  echo "---------------------------------------------------------------"
  echo "Checking run switch for Round $ROUND..."

  if [ ! -f "$SWITCH_FILE" ]; then
    echo "STOPPING: Switch file '$SWITCH_FILE' does not exist."
    break
  fi

  if [ ! -s "$SWITCH_FILE" ]; then
    echo "STOPPING: Switch file '$SWITCH_FILE' is empty."
    break
  fi

  SEED=$RANDOM
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)

  echo "Status OK. Starting Round $ROUND with SEED=${SEED} at ${TIMESTAMP}"
  echo "---------------------------------------------------------------"

  declare -A JOB_PIDS
  declare -A JOB_LOGS
  declare -A JOB_STATUS
  PIDS=()

  for idx in "${!MODES[@]}"; do
    mode=${MODES[$idx]}
    gpu_index=$(( idx % GPU_COUNT ))
    gpu="${GPU_IDS[$gpu_index]}"

    job_dir="${OUTPUT_BASE}_${mode}_${TIMESTAMP}"
    wandb_name="deit_${mode}_seed${SEED}_${TIMESTAMP}"
    mkdir -p "$job_dir"

    echo "  -> Launching ${mode} on GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=${gpu} python3 main.py \
      --model ${MODEL} \
      --data-path ${DATA_PATH} \
      --data-set CIFAR \
      --batch-size ${BATCH_SIZE} \
      --input-size ${INPUT_SIZE} \
      --epochs ${EPOCHS} \
      --lr ${LR} \
      --output_dir ${job_dir} \
      --wandb_project ${WANDB_PROJECT} \
      --wandb_name ${wandb_name} \
      --mask_mode ${mode} \
      --N 2 --M 4 \
      --nm_layers "${NM_LAYERS}" 
      > "${job_dir}/train.log" 2>&1 &

    pid=$!
    PIDS+=($pid)
    JOB_PIDS["$mode"]=$pid
    JOB_LOGS["$mode"]="${job_dir}/train.log"
    JOB_STATUS["$mode"]="RUNNING"
    sleep 1
  done

  echo ""
  echo "Started ${#PIDS[@]} jobs for Round $ROUND. Monitoring..."

  while true; do
    if [[ -t 1 ]]; then clear; else echo "-------------------------------"; fi

    echo "=== Job monitor Round $ROUND (Seed: $SEED) ==="
    echo "Switch file check: $([ -s "$SWITCH_FILE" ] && echo "Active" || echo "Empty/Missing - Will stop after this round")"

    all_done=true
    for mode in "${MODES[@]}"; do
      pid=${JOB_PIDS[$mode]}
      log=${JOB_LOGS[$mode]}
      status="UNKNOWN"

      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        status="RUNNING"
        all_done=false
      else
        if [ -f "$log" ]; then
          if tail -n 200 "$log" | grep -iE "traceback|error|exception" >/dev/null 2>&1; then
            status="FAILED"
          else
            status="EXITED"
          fi
        else
          status="NO_LOG"
        fi
      fi
      JOB_STATUS["$mode"]=$status

      printf "\n[%s] pid=%s status=%s log=%s\n" "$mode" "${pid:-N/A}" "${status}" "${log}"
      if [ -f "$log" ]; then
        tail -n 5 "$log" 2>/dev/null | sed "s/^/[${mode}] /"
      else
        echo "[${mode}] (no log yet)"
      fi
    done

    if $all_done; then
      break
    fi

    sleep 2
  done

  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  echo ""
  echo "Round $ROUND finished."
  echo "Waiting 5 seconds before checking switch file for next round..."
  sleep 5

done

echo "Script Execution Ended."
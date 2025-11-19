#!/bin/bash
# 并行启动多个 mask_mode 的测试/训练（自动填充 job_dir / wandb 名称）
# 新增功能: 自动创建并检查 run_switch.txt，文件为空或不存在时停止下一轮训练
set -e

# ---------- 可改的默认参数 ----------
ARCH="vgg19_cifar100"
DATA_PATH="/home/yzy/GitHub/Bi-Mask/datasets"
DATASET_NAME="cifar100"
LR="0.1"
WEIGHT_DECAY="0.001"
LABEL_SMOOTHING="0.1"
NUM_EPOCHS="300"
NM_LAYERS="features.30,features.33,features.36,features.40,features.43,features.46,features.49"

BASE_JOB_DIR="experiments/cifar100_${ARCH}"
BASE_PROJECT="bimask_cnn_${ARCH}_new"
BASE_WANDB="${ARCH}"

# Switch File Configuration
SWITCH_FILE="run_switch.txt"

# 要运行的 mask 模式（m1 与 m2/m3/m5 并行）
MODES=("m1" "m2" "m3" "m4" "m5")

# 解析 GPU 列表
if [ -n "$1" ]; then
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

# -------------------------------------------------------
# 1. 自动创建 Switch File (如果不存在)
# -------------------------------------------------------
if [ ! -f "$SWITCH_FILE" ]; then
  echo "ON" > "$SWITCH_FILE"
  echo "Created switch file: $SWITCH_FILE (Content: ON)"
fi

# 捕捉 Ctrl-C / 终止信号
function on_exit {
  echo ""
  echo "Caught signal, terminating all jobs..."
  # PIDS 必须是全局可见的
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

# =======================================================
# 主循环：不断开始新的训练 Set (直到 Switch File 变空/消失)
# =======================================================
ROUND=0
while true; do
  ROUND=$((ROUND + 1))
  
  # -----------------------------------------------------
  # 2. 检查 Switch File
  # -----------------------------------------------------
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

  # 如果文件存在且有内容，继续生成随机种子
  SEED=$RANDOM
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  
  echo "Status OK. Starting Round $ROUND with SEED=${SEED} at ${TIMESTAMP}"
  echo "---------------------------------------------------------------"

  # 重置状态变量
  declare -A JOB_PIDS
  declare -A JOB_LOGS
  declare -A JOB_STATUS
  PIDS=()

  # 启动当前轮次的所有作业
  for idx in "${!MODES[@]}"; do
    mode=${MODES[$idx]}
    gpu_index=$(( idx % GPU_COUNT ))
    gpu="${GPU_IDS[$gpu_index]}"

    job_dir="${BASE_JOB_DIR}_${mode}_${TIMESTAMP}"
    wandb_project="${BASE_PROJECT}"
    wandb_name="${BASE_WANDB}_${mode}_seed${SEED}_${TIMESTAMP}"

    mkdir -p "${job_dir}"

    echo "  -> Launching ${mode} on GPU ${gpu} (PID pending)"

    CUDA_VISIBLE_DEVICES=${gpu} python3 cifar.py \
      --arch ${ARCH} \
      --dataset ${DATASET_NAME} \
      --lr ${LR} \
      --gpus 0 \
      --weight_decay ${WEIGHT_DECAY} \
      --data_path ${DATA_PATH} \
      --label_smoothing ${LABEL_SMOOTHING} \
      --num_epochs ${NUM_EPOCHS} \
      --job_dir ${job_dir} \
      --mask_mode ${mode} \
      --wandb_project ${wandb_project} \
      --nm_layers ${NM_LAYERS} \
      --seed ${SEED} \
      --wandb_name ${wandb_name} > "${job_dir}/train.log" 2>&1 &

    pid=$!
    PIDS+=($pid)
    JOB_PIDS["$mode"]=$pid
    JOB_LOGS["$mode"]="${job_dir}/train.log"
    JOB_STATUS["$mode"]="RUNNING"
  done

  echo ""
  echo "Started ${#PIDS[@]} jobs for Round $ROUND. Monitoring..."

  # 监控循环 (当前轮次)
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

    # 检查本轮是否结束
    if $all_done; then
      break
    fi

    sleep 2
  done

  # 确保本轮所有进程完全退出
  for pid in "${PIDS[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  echo ""
  echo "Round $ROUND finished."
  echo "Waiting 5 seconds before checking switch file for next round..."
  sleep 5

done # 结束 while true 循环

echo "Script Execution Ended."
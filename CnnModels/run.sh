#!/bin/bash
# 并行启动多个 mask_mode 的测试/训练（自动填充 job_dir / wandb 名称）
# 用法:
#   ./run.sh              # 自动检测可用 GPU 并分配
#   ./run.sh 0,1,2,3      # 指定要使用的 GPU 列表（逗号分隔）
set -e

# ---------- 可改的默认参数 ----------
ARCH="resnet32_cifar100"
DATA_PATH="/home/yzy/GitHub/Bi-Mask/datasets"
DATASET_NAME="cifar100"
LR="0.1"
WEIGHT_DECAY="0.001"
LABEL_SMOOTHING="0.1"
NUM_EPOCHS="300"
NM_LAYERS="layer3"
SEED="114514"

# 要运行的 mask 模式（m1 与 m2/m3/m5 并行）
MODES=("m1" "m2" "m3" "m4" "m5")

# 解析 GPU 列表（如果传入参数则使用，否则尝试用 nvidia-smi 检测）
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

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_JOB_DIR="experiments/cifar100_${ARCH}"
BASE_PROJECT="bimask_cnn_${ARCH}"
BASE_WANDB="${ARCH}"

echo "Detected GPUs: ${GPU_IDS[*]}"
echo "Modes to run: ${MODES[*]}"
echo "Base job dir: ${BASE_JOB_DIR}"
echo "WandB base project: ${BASE_PROJECT}"
echo "Timestamp: ${TIMESTAMP}"
echo ""

# 映射：mode -> pid / log / status
declare -A JOB_PIDS
declare -A JOB_LOGS
declare -A JOB_STATUS
PIDS=()

# 捕捉 Ctrl-C / 终止信号，优雅杀死所有子进程
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

# 启动所有作业
for idx in "${!MODES[@]}"; do
  mode=${MODES[$idx]}
  # 轮询分配 GPU（尽量把不同模式分到不同卡）
  gpu_index=$(( idx % GPU_COUNT ))
  gpu="${GPU_IDS[$gpu_index]}"

  # 为每个模式生成独立的 job_dir / project / wandb name
  job_dir="${BASE_JOB_DIR}_${mode}_${TIMESTAMP}"
  wandb_project="${BASE_PROJECT}"
  wandb_name="${BASE_WANDB}_${mode}_seed${SEED}_${TIMESTAMP}"

  mkdir -p "${job_dir}"

  echo "Launching mode=${mode} on GPU ${gpu} -> job_dir=${job_dir}, wandb_project=${wandb_project}, wandb_name=${wandb_name}"

  # 使用 CUDA_VISIBLE_DEVICES 指定物理卡，并让程序内部的 --gpus 指向 0（可见设备中的第一个）
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
echo "Started ${#PIDS[@]} jobs. PIDs: ${PIDS[*]}"
echo "Logs: each job writes to its job_dir/train.log"
echo "Monitoring jobs in terminal. Press Ctrl-C to terminate all."

# 监控并在终端显示每个任务的最新日志（tail）和状态
while true; do
  # 如果终端支持，清屏；否则换行分隔
  if [[ -t 1 ]]; then
    clear
  else
    echo "-------------------------------"
  fi

  echo "=== Job monitor ($(date +'%Y-%m-%d %H:%M:%S')) ==="
  all_done=true
  for mode in "${MODES[@]}"; do
    pid=${JOB_PIDS[$mode]}
    log=${JOB_LOGS[$mode]}
    status="UNKNOWN"

    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      status="RUNNING"
      all_done=false
    else
      # 进程已退出或不存在
      if [ -f "$log" ]; then
        # 通过简单关键字判断是否异常退出（可根据需要扩展）
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

    # 输出模式头信息
    printf "\n[%s] pid=%s status=%s log=%s\n" "$mode" "${pid:-N/A}" "${status}" "${log}"

    # 输出最后 5 行日志（若存在），并在每行前加模式前缀
    if [ -f "$log" ]; then
      tail -n 5 "$log" 2>/dev/null | sed "s/^/[${mode}] /"
    else
      echo "[${mode}] (no log yet)"
    fi
  done

  echo -e "\nPress Ctrl-C to kill all running jobs."

  # 如果所有任务都已退出/失败，结束监控
  if $all_done; then
    break
  fi

  sleep 2
done

# 等待所有子进程退出（确保收集到退出码）
for pid in "${PIDS[@]}"; do
  wait "$pid" 2>/dev/null || true
done

echo ""
echo "All jobs finished. Final statuses:"
for mode in "${MODES[@]}"; do
  echo "  ${mode}: ${JOB_STATUS[$mode]}"
done
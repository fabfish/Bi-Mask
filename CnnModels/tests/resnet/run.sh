#!/bin/bash
# ResNet32 CIFAR-10 并行测试运行脚本
# 用法：cd CnnModels/tests/resnet && ./run.sh
# 本脚本根据config.sh中定义的测试用例，自动在各GPU上并行运行训练任务

set -e

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 导入配置文件
source "$SCRIPT_DIR/config.sh"

# 切换到CnnModels目录（确保cifar.py和相关文件可访问）
cd "$SCRIPT_DIR/../../"

# 自动生成 WANDB_PROJECT（如果未指定）
# 格式：bimask_cnn_${MODEL_NAME}_${DATASET}
if [ -z "$WANDB_PROJECT" ]; then
    MODEL_PREFIX="${ARCH%_*}"  # 提取模型名称（如 resnet32）
    WANDB_PROJECT="bimask_cnn_${MODEL_PREFIX}_${DATASET}"
fi

echo "========================================"
echo "ResNet32 CIFAR-10 并行测试启动"
echo "========================================"
echo "配置信息："
echo "  - 架构: $ARCH"
echo "  - 数据集: $DATASET"
echo "  - 数据路径: $DATA_PATH"
echo "  - NM Layers: $NM_LAYERS"
echo "  - 总测试用例数: ${#TEST_CASES[@]}"
echo "========================================"

# 创建实验输出目录
mkdir -p "$EXPERIMENTS_DIR"

# 存储所有后台进程PID
declare -a PIDS=()
declare -a TASK_NAMES=()

# 遍历所有测试用例，启动后台任务
for i in "${!TEST_CASES[@]}"; do
    IFS='|' read -r GPU_ID MASK_MODE <<< "${TEST_CASES[$i]}"
    
    # 自动生成 experiment name
    # 格式：${MODEL_NAME}_${DATASET}_${MASK_MODE}_rand31_${NM_LAYERS_ESCAPED}_sd${SEED}
    # 例如：resnet32_cifar10_m2_rand31_layer3_sd24
    MODEL_PREFIX="${ARCH%_*}"
    DATASET_SUFFIX="${ARCH##*_}"
    NM_LAYERS_ESCAPED="${NM_LAYERS//./}"  # 去除点号以避免路径问题
    TASK_NAME="${MODEL_PREFIX}_${DATASET_SUFFIX}_${MASK_MODE}_rand31_${NM_LAYERS_ESCAPED}_sd${SEED}"
    # wandb_name 使用同样的格式，确保与本地任务名一致
    WANDB_NAME="$TASK_NAME"
    
    # 构建输出目录名称
    JOB_DIR="${EXPERIMENTS_DIR}/${TASK_NAME}"
    
    echo ""
    echo "[任务 $((i+1))/${#TEST_CASES[@]}] 启动: $TASK_NAME"
    echo "  GPU: $GPU_ID, Mask Mode: $MASK_MODE"
    
    # 后台启动训练任务
    # CUDA_VISIBLE_DEVICES 限制只能访问指定的GPU
    # &> 将stdout和stderr重定向到日志文件
    (
        CUDA_VISIBLE_DEVICES=$GPU_ID python3 cifar.py \
            --arch "$ARCH" \
            --lr "$LR" \
            --gpus "$GPU_ID" \
            --weight_decay "$WEIGHT_DECAY" \
            --data_path "$DATA_PATH" \
            --label_smoothing "$LABEL_SMOOTHING" \
            --num_epochs "$NUM_EPOCHS" \
            --job_dir "$JOB_DIR" \
            --mask_mode "$MASK_MODE" \
            --wandb_project "$WANDB_PROJECT" \
            --nm_layers "$NM_LAYERS" \
            --seed "$SEED" \
            --wandb_name "$WANDB_NAME"
    ) &> "${JOB_DIR}.log" &
    
    # 保存进程PID和任务名称
    PIDS+=($!)
    TASK_NAMES+=("$TASK_NAME")
    
    # 短暂延迟以避免启动过快
    sleep 1
done

echo ""
echo "========================================"
echo "所有任务已在后台启动"
echo "========================================"

# 等待所有后台任务完成，并监控进程状态
FAILED_TASKS=()
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    TASK=${TASK_NAMES[$i]}
    
    # 等待进程结束并获取返回值
    if wait $PID; then
        echo "✓ 任务成功: $TASK (PID: $PID)"
    else
        echo "✗ 任务失败: $TASK (PID: $PID)"
        FAILED_TASKS+=("$TASK")
    fi
done

echo ""
echo "========================================"
echo "所有任务执行完成"
echo "========================================"

# 输出测试结果摘要
if [ ${#FAILED_TASKS[@]} -eq 0 ]; then
    echo "✓ 所有任务执行成功！"
    echo "日志文件位置: ${EXPERIMENTS_DIR}/*.log"
    exit 0
else
    echo "✗ 以下任务执行失败："
    for task in "${FAILED_TASKS[@]}"; do
        echo "  - $task"
    done
    echo "请查看对应的日志文件进行诊断"
    exit 1
fi

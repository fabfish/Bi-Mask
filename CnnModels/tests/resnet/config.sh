#!/bin/bash
# ResNet32 CIFAR-10 测试配置文件
# 本文件仅需定义基础参数，experiment name自动生成

# ==================== 基础模型参数 ====================
ARCH="resnet32_cifar10"
DATASET="cifar10"
DATA_PATH="/root/Bi-Mask/datasets"

# ==================== 训练超参数 ====================
NUM_EPOCHS=300
LR=0.1
WEIGHT_DECAY=0.001
LABEL_SMOOTHING=0.1
NM_LAYERS="layer3"
SEED=24

# ==================== wandb 配置 ====================
# wandb_project 由 run.sh 自动生成：bimask_cnn_${MODEL}_${DATASET}
# 例如：bimask_cnn_resnet_cifar10
WANDB_PROJECT=""  # 留空，由run.sh自动填充

# ==================== 测试用例定义 ====================
# 仅需指定：GPU编号和mask_mode
# 格式：GPU_ID|MASK_MODE
# experiment name 由脚本自动生成：
#   ${ARCH%_*}/${ARCH##*_}_${MASK_MODE}_rand31_${NM_LAYERS}_sd${SEED}
declare -a TEST_CASES=(
    "0|m2"
    "1|m3"
    "2|m4"
    "3|m5"
)

# ==================== 输出目录 ====================
EXPERIMENTS_DIR="experiments"

# ==================== 并发控制 ====================
# 最大并发任务数（根据可用GPU数调整）
MAX_PARALLEL_JOBS=4

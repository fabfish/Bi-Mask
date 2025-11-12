#!/bin/bash
# 测试框架说明
# 
# 文件结构:
#   CnnModels/tests/
#   ├── run_all.sh           # 主运行脚本 (在tests目录下执行)
#   ├── resnet/
#   │   ├── config.sh        # ResNet配置文件 (定义所有测试用例)
#   │   └── run.sh           # ResNet运行脚本
#   └── vgg/
#       ├── config.sh        # VGG配置文件 (定义所有测试用例)
#       └── run.sh           # VGG运行脚本
#
# ========== 使用方法 ==========
#
# 1. 运行单个测试组 (例如 ResNet):
#    cd CnnModels/tests/resnet
#    bash run.sh
#
# 2. 运行单个测试组 (例如 VGG):
#    cd CnnModels/tests/vgg
#    bash run.sh
#
# 3. 并行运行所有测试组:
#    cd CnnModels/tests
#    bash run_all.sh
#
# ========== 修改配置 ==========
#
# 1. 添加新的GPU/任务到ResNet测试:
#    - 编辑 CnnModels/tests/resnet/config.sh
#    - 在 TEST_CASES 数组中添加新项
#    - 格式: "GPU_ID|MASK_MODE" (experiment name 自动生成)
#
# 2. 修改ResNet的学习参数:
#    - 编辑 CnnModels/tests/resnet/config.sh
#    - 修改 LR, WEIGHT_DECAY, NUM_EPOCHS 等变量
#
# 3. 修改数据路径:
#    - 编辑相应目录的 config.sh
#    - 修改 DATA_PATH 变量
#
# 4. 自定义实验名称前缀:
#    - run.sh 自动生成 experiment name: MODEL_DATASET_MASKMODE_rand31_LAYER_sdSEED
#    - 要修改格式，编辑相应 run.sh 中的 TASK_NAME 生成逻辑
#
# 5. 新增测试模型 (例如 MobileNet):
#    - 创建目录: mkdir -p CnnModels/tests/mobilenet
#    - 复制 resnet/config.sh 和 resnet/run.sh
#    - 修改 config.sh 中的参数
#    - run_all.sh 会自动发现并运行
#
# ========== 实现细节 ==========
#
# 并行执行逻辑:
# - 每个 TEST_CASE 在指定的GPU上独立运行
# - CUDA_VISIBLE_DEVICES 限制进程只能访问指定的GPU
# - 使用后台进程(&)实现并行
# - run.sh 脚本等待所有子进程完成后才退出
# - run_all.sh 脚本并行启动多个测试组
#
# GPU分配:
# - config.sh 中 TEST_CASES 数组的第一列是 GPU_ID
# - 若有4张GPU (0,1,2,3)，可同时运行4个任务
# - GPU_ID 必须是有效的GPU索引
# - CUDA_VISIBLE_DEVICES 确保进程绑定到正确的GPU
#
# 日志管理:
# - 每个任务的输出重定向到 ${JOB_DIR}.log
# - run_all.sh 的日志存储在 tests/ 目录下，格式: ${test_group}_group.log
# - 使用 tail -f 命令可实时查看日志
# - 失败任务的日志位置会在最终摘要中显示
#
# 返回值处理:
# - 所有任务成功则返回 0
# - 如果任何任务失败则返回 1
# - 使用 $? 可检查脚本执行结果
#
# ========== 常见命令 ==========
#
# # 查看实时日志
# tail -f CnnModels/tests/resnet/experiments/*.log
#
# # 查看所有GPU上运行的进程
# nvidia-smi
#
# # 计算运行时间
# time bash run.sh
#
# # 在后台运行并关闭终端后继续执行
# nohup bash run.sh > output.log 2>&1 &
#
# # 杀死所有python进程
# pkill -f cifar.py
#
# ========== 参考配置 ==========
#
# TEST_CASES 示例 - 4 GPU 场景:
# declare -a TEST_CASES=(
#     "0|m2"  # GPU 0 运行 m2 模式，自动生成名称
#     "1|m2"  # GPU 1 运行 m2 模式
#     "2|m3"  # GPU 2 运行 m3 模式
#     "3|m3"  # GPU 3 运行 m3 模式
# )
#
# TEST_CASES 示例 - 2 GPU 场景（需要顺序执行或分批）:
# declare -a TEST_CASES=(
#     "0|m2"  # GPU 0 运行 m2 模式
#     "1|m3"  # GPU 1 运行 m3 模式
# )
#
# 自动生成的 experiment name 格式:
# ${MODEL_PREFIX}_${DATASET}_${MASK_MODE}_rand31_${NM_LAYERS_ESCAPED}_sd${SEED}
# 例如:
#   - resnet32_cifar10_m2_rand31_layer3_sd24
#   - vgg_cifar10_m2_rand31_features49_sd24

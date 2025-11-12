# 快速开始

## 运行测试

```bash
# 进入测试目录
cd CnnModels/tests

# 运行所有测试（并行执行 resnet 和 vgg）
bash run_all.sh

# 或只运行单个测试组
cd resnet && bash run.sh
cd vgg && bash run.sh
```

## 添加/修改测试用例

### 添加新的测试任务
编辑 `resnet/config.sh` 或 `vgg/config.sh`，在 `TEST_CASES` 数组中添加：

```bash
# 格式：GPU_ID|MASK_MODE
declare -a TEST_CASES=(
    "0|m2"    # GPU 0，m2 模式
    "1|m3"    # GPU 1，m3 模式
    "2|m4"    # GPU 2，m4 模式
)
```

**experiment name 自动生成**，无需手动指定。

### 修改训练参数
编辑相应的 `config.sh`：

```bash
ARCH="resnet32_cifar10"       # 改模型
DATA_PATH="/path/to/data"     # 改数据路径
NUM_EPOCHS=300                # 改轮数
LR=0.1                        # 改学习率
NM_LAYERS="layer3"            # 改 nm layers
```

## 示例：自动生成的 experiment name

| 配置 | 生成的名称 |
|------|-----------|
| ResNet, m2 mode, layer3 | `resnet32_cifar10_m2_rand31_layer3_sd24` |
| VGG, m3 mode, features.49 | `vgg_cifar10_m3_rand31_features49_sd24` |

## 新增模型目录

```bash
# 创建新模型目录（如 mobilenet）
mkdir -p CnnModels/tests/mobilenet

# 复制配置文件
cp CnnModels/tests/resnet/config.sh CnnModels/tests/mobilenet/
cp CnnModels/tests/resnet/run.sh CnnModels/tests/mobilenet/

# 修改 config.sh 中的参数即可
# run_all.sh 会自动发现并运行
```

## 监控和调试

```bash
# 查看实时日志
tail -f CnnModels/tests/resnet/experiments/*.log

# 查看 GPU 使用情况
nvidia-smi

# 查看后台任务
ps aux | grep cifar.py

# 杀死所有训练任务
pkill -f cifar.py
```

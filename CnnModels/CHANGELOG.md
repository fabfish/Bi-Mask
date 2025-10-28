# CnnModels 更新总结

## 修复的问题

### 1. Mask维度不匹配错误修复

**问题描述**：
- `post_mask_apply()` 方法中 `self.weight.data *= self.forward_mask.t()` 出现维度不匹配错误
- 错误信息：`RuntimeError: The size of tensor a (3) must match the size of tensor b (27) at non-singleton dimension 3`

**修复方案**：
- 在 `post_mask_apply()` 方法中正确处理权重和mask的维度转换
- 使用 `get_n_m_sparse_matrix()` 函数来生成正确的稀疏权重
- 确保权重形状与原始卷积层权重形状一致

### 2. GPU选择问题修复

**问题描述**：
- 无论选择哪个GPU（如 `--gpus 1` 或 `--gpus 2`），都显示 `cuda:0`
- 这是因为设置 `CUDA_VISIBLE_DEVICES` 后，PyTorch重新映射了GPU索引

**修复方案**：
- 在设置 `CUDA_VISIBLE_DEVICES` 后，正确重新映射GPU索引
- 添加调试信息显示原始GPU选择和重新映射后的索引
- 确保设备选择使用正确的GPU索引

## 新增功能

### 1. Random Mask 支持

**功能描述**：
- 添加了random mask功能，作为N:M半结构化mask的替代方案
- 使用topk算法基于权重的绝对值选择最重要的元素
- 支持自定义稀疏度比例（默认50%）

**新增参数**：
- `--use_random_mask`: 启用random mask（默认: False）
- `--random_mask_ratio`: 保留元素的比例（默认: 0.5，即50%稀疏度）

**实现细节**：
- 在CNN和DeiT中都添加了random mask支持
- 实现了`get_random_sparse_matrix()`函数，使用topk算法
- 与现有的mask模式（m2, m3, m4）完全兼容
- 在模型配置输出中显示mask类型（N:M或random）

**使用方法**：
```bash
# 使用random mask（50%稀疏度）
python cifar.py --arch resnet32_cifar10 --use_random_mask --random_mask_ratio 0.5

# 使用random mask（30%稀疏度）
python cifar.py --arch resnet32_cifar10 --use_random_mask --random_mask_ratio 0.3

# DeiT中使用random mask
python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model vit_deit_small_patch16_224 --use_random_mask --random_mask_ratio 0.5
```

### 2. Mask Mode 配置输出

在模型训练开始时，会自动输出每一层的 mask mode 配置信息：

```
=== Model Mask Mode Configuration ===
Global mask_mode: m2
Layer-wise mask modes:
  Layer 1: conv1 -> mask_mode: m2
  Layer 2: layer1.0.conv1 -> mask_mode: m2
  Layer 3: layer1.0.conv2 -> mask_mode: m2
  ...
Total NMConv layers: 25
========================================
```

这样可以确认模型运行时的运算模式是否正确。

### 2. Wandb 日志记录支持

参考 DeiT 中的 wandb 实现，为 CnnModels 添加了完整的 wandb 支持：

#### 新增参数
- `--wandb_project`: wandb 项目名称（默认: 'bimask_cnn'）
- `--wandb_name`: wandb 运行名称（可选，会自动生成）

#### 实现细节
- 在 `utils/options.py` 中添加了 wandb 相关参数
- 在 `cifar.py` 和 `imagenet.py` 中添加了 wandb 初始化和日志记录
- 支持训练和验证指标的实时记录
- 自动生成有意义的运行名称（包含模型、mask模式、参数等信息）

#### 使用方法
```bash
# CIFAR-10 训练示例
python cifar.py --arch resnet32_cifar10 --lr 0.1 --data_path PATH_TO_DATASETS --num_epochs 300 --job_dir PATH_TO_JOB_DIR --wandb_project bimask_cnn --wandb_name resnet32_m2

# ImageNet 训练示例  
python imagenet.py --arch resnet50 --lr 0.1 --data_path PATH_TO_DATASETS --num_epochs 120 --job_dir PATH_TO_JOB_DIR --wandb_project bimask_cnn --wandb_name resnet50_m2
```

### 3. Bi-Mask 多种模式支持

参考 `vision_transformer.py` 中的实现，为 CnnModels 添加了不同的 Bi-Mask 模式：

#### 新增参数
- `--mask_mode`: Bi-Mask 模式选择（'m2', 'm3', 'm4'）

#### 模式说明
- **m2**: 双向 mask 模式（forward_mask + backward_mask）
- **m3**: Pre-mask 模式（在前向传播前应用 mask）
- **m4**: Post-mask 模式（在优化器步骤后应用 mask）

#### 实现细节
- 在 `utils/options.py` 中添加了 `--mask_mode` 参数
- 在 `utils/conv_type.py` 中修改了 `NMConv` 类：
  - 添加了 `mask_mode` 属性
  - 修改了 `forward` 方法以支持不同模式
  - m2 模式：使用双向 mask（原始实现）
  - m3 模式：Pre-mask 模式，在前向传播前应用 mask
  - m4 模式：Post-mask 模式，在优化器步骤后应用 mask
- 在 `cifar.py` 和 `imagenet.py` 中添加了 mask mode 配置输出功能

#### 使用方法
```bash
# 使用 m2 模式（双向 mask）
python cifar.py --arch resnet32_cifar10 --mask_mode m2 --wandb_project bimask_cnn

# 使用 m3 模式（仅前向 mask）
python cifar.py --arch resnet32_cifar10 --mask_mode m3 --wandb_project bimask_cnn

# 使用 m4 模式（默认）
python cifar.py --arch resnet32_cifar10 --mask_mode m4 --wandb_project bimask_cnn
```

## 修改的文件

1. **CnnModels/utils/options.py**
   - 添加 wandb 相关参数
   - 添加 mask_mode 参数
   - 添加 random mask 相关参数

2. **CnnModels/utils/conv_type.py**
   - 修改 NMConv 类以支持不同 mask 模式
   - 添加 mask_mode 属性
   - 修改 forward 方法实现不同模式
   - 添加 MyConv2d_Lay_m3 类用于 m3/m4 模式
   - 修复 post_mask_apply() 方法中的维度不匹配问题
   - 添加 random mask 支持（get_random_sparse_matrix函数）
   - 修改 NMConv 类支持 random mask

3. **CnnModels/cifar.py**
   - 添加 wandb 导入和初始化
   - 在训练和验证函数中添加 wandb 日志记录
   - 在主训练循环中添加 epoch 级别的日志记录
   - 添加 mask mode 配置输出功能
   - 添加 post-mask 应用功能
   - 修复 GPU 选择问题
   - 更新 mask mode 配置输出显示 random mask 信息

4. **CnnModels/imagenet.py**
   - 添加 wandb 导入和初始化
   - 在训练和验证函数中添加 wandb 日志记录
   - 在主训练循环中添加 epoch 级别的日志记录
   - 添加 mask mode 配置输出功能
   - 修复 GPU 选择问题
   - 更新 mask mode 配置输出显示 random mask 信息

5. **DeiT/main.py**
   - 添加 random mask 相关参数

6. **DeiT/timm/models/vision_transformer.py**
   - 添加 get_random_sparse_matrix 函数
   - 修改 NMConv 类支持 random mask
   - 更新 forward 和 post_mask_apply 方法

7. **README.md**
   - 更新文档说明新功能
   - 添加使用示例
   - 更新 mask mode 说明
   - 添加 random mask 使用说明

8. **CnnModels/example_usage.sh**
   - 创建示例脚本展示如何使用新功能

9. **CnnModels/random_mask_examples.sh**
   - 创建 random mask 使用示例脚本

10. **CnnModels/test_mask_mode.py**
    - 创建测试脚本演示 mask mode 输出功能

11. **CnnModels/test_fixes.py**
    - 创建测试脚本验证mask维度修复和GPU选择修复

12. **CnnModels/CHANGELOG.md**
    - 详细更新说明

## 兼容性

- 所有修改都向后兼容
- 如果不指定 wandb 参数，不会启用 wandb 日志记录
- 如果不指定 mask_mode，默认使用 m4 模式（与原始实现相同）
- 现有的训练脚本无需修改即可正常运行

## 安装要求

使用 wandb 功能需要安装 wandb：
```bash
pip install wandb
```

如果不安装 wandb，代码会显示警告但不会影响训练。

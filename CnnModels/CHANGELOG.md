# CnnModels 更新总结

## 新增功能

### 1. Wandb 日志记录支持

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

### 2. Bi-Mask 多种模式支持

参考 `vision_transformer.py` 中的实现，为 CnnModels 添加了不同的 Bi-Mask 模式：

#### 新增参数
- `--mask_mode`: Bi-Mask 模式选择（'m2', 'm3', 'm4'）

#### 模式说明
- **m2**: 双向 mask 模式（forward_mask + backward_mask）
- **m3**: 仅前向 mask 模式（只使用 forward_mask）
- **m4**: 默认模式（与原始实现相同）

#### 实现细节
- 在 `utils/options.py` 中添加了 `--mask_mode` 参数
- 在 `utils/conv_type.py` 中修改了 `NMConv` 类：
  - 添加了 `mask_mode` 属性
  - 修改了 `forward` 方法以支持不同模式
  - m2 模式：使用双向 mask（原始实现）
  - m3 模式：仅使用前向 mask，简化计算
  - m4 模式：默认模式（与原始相同）

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

2. **CnnModels/utils/conv_type.py**
   - 修改 NMConv 类以支持不同 mask 模式
   - 添加 mask_mode 属性
   - 修改 forward 方法实现不同模式

3. **CnnModels/cifar.py**
   - 添加 wandb 导入和初始化
   - 在训练和验证函数中添加 wandb 日志记录
   - 在主训练循环中添加 epoch 级别的日志记录

4. **CnnModels/imagenet.py**
   - 添加 wandb 导入和初始化
   - 在训练和验证函数中添加 wandb 日志记录
   - 在主训练循环中添加 epoch 级别的日志记录

5. **README.md**
   - 更新文档说明新功能
   - 添加使用示例

6. **CnnModels/example_usage.sh**
   - 创建示例脚本展示如何使用新功能

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

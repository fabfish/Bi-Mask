# Bi-directional Masks for Efficient N:M Sparse Training (ICML 2023)

**论文链接 (Paper Link):** [https://arxiv.org/abs/2302.06058](https://arxiv.org/abs/2302.06058)

## 🛠️ 环境要求

- python 3.7
- pytorch 1.10.2
- torchvision 0.11.3

## 🚀 环境安装

1. **创建 Conda 环境:**

    ```bash
    conda create -n bimask python=3.7
    conda activate bimask
    ```

2. **安装 PyTorch 和 Torchvision:**
    请根据您的 CUDA 版本选择合适的命令。以下是几个示例：

      - **For CUDA 10.2 (例如 V100):**

        ```bash
        pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html
        ```

      - **For CUDA 11.1 (例如 A100):**

        ```bash
        pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html
        ```

      - **For CUDA 11.3:**

        ```bash
        pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
        ```

    > **注意:** 如果通过 `conda install` 安装，可能默认安装 CPU 版本的 PyTorch，建议使用 `pip` 指定 CUDA 版本进行安装。

-----

## 🏃‍♂️ 模型训练

### 1\. 在 ImageNet 上训练

- **ResNet-18**

    ```bash
    cd CnnModels
    python imagenet.py --arch resnet18 --lr 0.1 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 120 --job_dir PATH_TO_JOB_DIR --iter 100 --greedy_num 100
    ```

- **ResNet-50**

    ```bash
    cd CnnModels
    python imagenet.py --arch resnet50 --lr 0.1 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 120 --job_dir PATH_TO_JOB_DIR --iter 100 --greedy_num 100
    ```

- **DeiT-small**

    ```bash
    cd DeiT
    python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path PATH_TO_DATASETS --output_dir PATH_TO_JOB_DIR
    ```

### 2\. 在 CIFAR 上训练

- **VGG-19**

    ```bash
    cd CnnModels
    python cifar.py --arch vgg19_cifar10 --lr 0.1 --weight_decay 0.001 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 300 --job_dir PATH_TO_JOB_DIR
    ```

- **ResNet-32**

    ```bash
    cd CnnModels
    python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 300 --job_dir PATH_TO_JOB_DIR
    ```

- **MobileNetV2**

    ```bash
    cd CnnModels
    python cifar.py --arch mobilenetv2 --lr 0.1 --weight_decay 0.001 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 300 --job_dir PATH_TO_JOB_DIR
    ```

-----

## ✨ 高级功能 (CnnModels)

`CnnModels` 文件夹下的脚本支持 Wandb 日志、不同的 Bi-Mask 模式和随机掩码。

### 1\. Wandb 日志

添加 `--wandb_project` 和 `--wandb_name` 参数来启用 wandb 日志记录：

```bash
cd CnnModels
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 \
  --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 300 \
  --job_dir PATH_TO_JOB_DIR --wandb_project bimask_cnn --wandb_name resnet32_m2
```

### 2\. Bi-Mask 模式

使用 `--mask_mode` 参数选择不同的 Bi-Mask 实现：

- `m2`: 默认的双向掩码 (forward + backward mask)
- `m3`: Pre-mask 模式 (在 forward pass 之前应用 mask)
- `m4`: Post-mask 模式 (在 optimizer step 之后应用 mask)

模型将在训练开始时自动打印每一层的掩码模式配置。

### 3\. 随机掩码支持

使用 `--use_random_mask` 标志来启用随机掩码，以替代 N:M 半结构化掩码。

- `--use_random_mask`: 启用随机掩码 (默认为 `False`)
- `--random_mask_ratio`: 随机掩码保留的元素比例 (默认为 `0.5`，即 50% 稀疏度)

随机掩码使用 topk 算法，根据权重的绝对值来选择最重要的元素。

**示例：**

```bash
# 示例 1: 训练 N:M 掩码 (2:4 模式)
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 \
  --data_path PATH_TO_DATASETS --num_epochs 300 --job_dir PATH_TO_JOB_DIR \
  --mask_mode m2 --wandb_project bimask_cnn

# 示例 2: 训练随机掩码 (50% 稀疏度)
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 \
  --data_path PATH_TO_DATASETS --num_epochs 300 --job_dir PATH_TO_JOB_DIR \
  --use_random_mask --random_mask_ratio 0.5 --wandb_project bimask_cnn

# 示例 3: 训练随机掩码 (30% 稀疏度)
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 \
  --data_path PATH_TO_DATASETS --num_epochs 300 --job_dir PATH_TO_JOB_DIR \
  --use_random_mask --random_mask_ratio 0.3 --wandb_project bimask_cnn
```

-----

## 📊 测试与预训练模型

### 1\. 预训练模型

我们提供了训练好的模型和实验日志：

| Model | Sparse Pattern | Top1 | Top5 | Link |
| :--- | :--- | :--- | :--- | :--- |
| ResNet-50 | 2:4 | 77.4 | 93.7 | [Google Drive](https://drive.google.com/drive/folders/1LvUQe1TOhEYE9HF4D9YEOF1uyid8JdlX?usp=share_link) |
| ResNet-50 | 1:4 | 75.6 | 92.7 | [Google Drive](https://drive.google.com/drive/folders/1IVOJFmKIq--hOuZs5fhz2GZT5QY17XCg?usp=share_link) |
| ResNet-50 | 2:8 | 76.3 | 93.0 | [Google Drive](https://drive.google.com/drive/folders/1nlUf5D1sEV48z1I3H5zZp03GVhI-K9-l?usp=share_link) |
| ResNet-50 | 4:8 | 77.5 | 93.8 | [Google Drive](https://drive.google.com/drive/folders/1hlWULurqYExy8sImJTXtAcf9CMEiVJoI?usp=share_link) |
| ResNet-50 | 1:16 | 71.4 | 90.1 | [Google Drive](https://drive.google.com/drive/folders/1LxHqcmN2buPTFuP_QawYre92dx9b8CFe?usp=share_link) |
| Deit-small | 2:4 | 77.6 | 93.8 | [Google Drive](https://drive.google.com/drive/folders/11auZ08_OgPnebfSF7Fp7ASB7YsNcrjZa?usp=sharing) |

### 2\. 测试指令

- **ResNet-50 on ImageNet**

    ```bash
    cd CnnModels
    python eval.py --arch resnet50 --pretrain_dir PATH_TO_CHECKPOINTS --train_batch_size 256 --eval_batch_size 256 --label_smoothing 0.1 --data_path PATH_TO_DATASETS
    ```

- **DeiT-small on ImageNet**

    ```bash
    cd DeiT
    python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path PATH_TO_DATASETS --output_dir PATH_TO_JOB_DIR --resume PATH_TO_CHECKPOINTS --eval
    ```

-----

## 📚 附录：开发者笔记

以下是原始文档中包含的特定环境配置、数据准备脚本和运行命令，仅供参考。

### 1\. ImageNet 数据准备 (示例)

```bash
# 1. 下载或同步数据集（以 hetao new 环境为例）
cd ~
rsync -avP /data/lishen/yzy/ILSVRC2012_img_train.tar ./
rsync -avP /data/lishen/yzy/ILSVRC2012_img_val.tar ./
cp /data/lishen/yzy/valprep.sh val/

# 2. 创建 train 目录并解压
mkdir -p imagenet/train
cd imagenet
tar -xvf ../ILSVRC2012_img_train.tar -C train
cd train
# 解压各个子 tar 包
for x in `ls *.tar`; do
  fn=`basename $x .tar`
  mkdir $fn
  tar -xvf $x -C $fn
  rm -f $x
done

# 3. 创建 val 目录并解压
cd ~/imagenet
mkdir val
tar -xvf ../ILSVRC2012_img_val.tar -C val
cd val
bash valprep.sh # 运行验证集处理脚本
```

### 2\. 特定运行命令示例

这些命令包含了特定于机器的路径和配置。

- **Hetao (DeiT 训练)**

    ```bash
    cd DeiT
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
      --model vit_deit_small_patch16_224 --batch-size 256 \
      --data-path /data/datasets/ImageNet1k --output_dir /data/yzy/bimask/deit_imagenet
    ```

- **Hetao (DeiT 评估)**

    ```bash
    cd DeiT
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
      --model vit_deit_small_patch16_224 --batch-size 256 \
      --data-path /data/datasets/ImageNet1k --output_dir /data/yzy/bimask/deit_imagenet \
      --eval
    ```

- **Hetao New (DeiT 训练)**

    ```bash
    # 假定已 clone 仓库: git clone git@github.com:fabfish/Bi-Mask.git
    cd /root/Bi-Mask/DeiT
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --use_env main.py \
      --model vit_deit_small_patch16_224 --batch-size 256 \
      --data-path /root/imagenet --output_dir /root/deit_imagenet \
      --num_workers 0 --no-pin-mem
    ```

- **Hetao New (CIFAR 训练)**

    ```bash
    cd /root/Bi-Mask/CnnModels
    python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 \
      --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 300 \
      --job_dir /root/resnet32_cifar10_test
    ```

### 3\. 其他命令

- **终止进程**

    ```bash
    sudo pkill -f "cifar.py"
    ```

- **设置 pip 镜像源**

    ```bash
    pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
    ```

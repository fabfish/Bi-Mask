# Bi-directional Masks for Efficient N:M Sparse Training (ICML 2023) ([Paper Link](https://arxiv.org/abs/2302.06058))

## Requirements

- python 3.7
- pytorch 1.10.2
- torchvision 0.11.3

## Training

### Training models on ImageNet

- ResNet-18

```bash
cd CnnModels
python imagenet.py --arch resnet18 --lr 0.1 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 120 --job_dir PATH_TO_JOB_DIR --iter 100 --greedy_num 100
```

- ResNet-50

```bash
cd CnnModels
python imagenet.py --arch resnet50 --lr 0.1 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 120 --job_dir PATH_TO_JOB_DIR --iter 100 --greedy_num 100
```

- DeiT-small

```bash
cd DeiT
python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path PATH_TO_DATASETS --output_dir PATH_TO_JOB_DIR
```

### Training models on CIFAR

- VGG-19

```bash
cd CnnModels
python cifar.py --arch vgg19_cifar10 --lr 0.1 --weight_decay 0.001 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 300 --job_dir PATH_TO_JOB_DIR
```

- ResNet-32

```bash
cd CnnModels
python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 300 --job_dir PATH_TO_JOB_DIR
```

- MobileNetV2

```bash
cd CnnModels
python cifar.py --arch mobilenetv2 --lr 0.1 --weight_decay 0.001 --data_path PATH_TO_DATASETS --label_smoothing 0.1 --num_epochs 300 --job_dir PATH_TO_JOB_DIR
```

## Testing

We provide our trained models and experiment logs at following Table:

| Model      | Sparse Pattern | Top1 | Top5 | Link                                                                                                 |
| ---------- | -------------- | ---- | ---- | ---------------------------------------------------------------------------------------------------- |
| ResNet-50  | 2:4            | 77.4 | 93.7 | [Google Drive](https://drive.google.com/drive/folders/1LvUQe1TOhEYE9HF4D9YEOF1uyid8JdlX?usp=share_link) |
| ResNet-50  | 1:4            | 75.6 | 92.7 | [Google Drive](https://drive.google.com/drive/folders/1IVOJFmKIq--hOuZs5fhz2GZT5QY17XCg?usp=share_link) |
| ResNet-50  | 2:8            | 76.3 | 93.0 | [Google Drive](https://drive.google.com/drive/folders/1nlUf5D1sEV48z1I3H5zZp03GVhI-K9-l?usp=share_link) |
| ResNet-50  | 4:8            | 77.5 | 93.8 | [Google Drive](https://drive.google.com/drive/folders/1hlWULurqYExy8sImJTXtAcf9CMEiVJoI?usp=share_link) |
| ResNet-50  | 1:16           | 71.4 | 90.1 | [Google Drive](https://drive.google.com/drive/folders/1LxHqcmN2buPTFuP_QawYre92dx9b8CFe?usp=share_link) |
| Deit-small | 2:4            | 77.6 | 93.8 | [Google Drive](https://drive.google.com/drive/folders/11auZ08_OgPnebfSF7Fp7ASB7YsNcrjZa?usp=sharing)    |

To test, run:

- ResNet-50 on ImageNet

```bash
cd CnnModels
python eval.py --arch resnet50 --pretrain_dir PATH_TO_CHECKPOINTS --train_batch_size 256 --eval_batch_size 256  --label_smoothing 0.1 --data_path PATH_TO_DATASETS
```

- DeiT-small on ImageNet

```bash
cd DeiT
python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path PATH_TO_DATASETS --output_dir PATH_TO_JOB_DIR --resume PATH_TO_CHECKPOINTS --eval
```

v100

conda create -n bimask python=3.7 cudatoolkit=10.2

(如果通过conda装到的可能会是cpu版本的torch，需要通过pip安装)

pip install torch==1.10.2+cu102 torchvision==0.11.3+cu102 -f https://download.pytorch.org/whl/cu102/torch_stable.html

sudo pkill -f "cifar.py"

a100

conda create -n bimask python=3.7

<!-- pip install torch==1.10.2 torchvision==0.11.3 -->


<!-- pip install torch==1.10.2 torchvision==0.11.3 -f https://download.pytorch.org/whl/cu124/torch_stable.html -->


<!-- conda install pytorch==1.10.2 torchvision==0.11.3 cudatoolkit=11.3 -c pytorch -c conda-forge -->

pip install torch==1.10.2+cu111 torchvision==0.11.3+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html


hetao

conda create -n bimask python=3.7 cudatoolkit=11.3

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path /hefhspace/yzy/imagenet-1k --output_dir /hefhspace/yzy/bimask/deit_imagenet


cd DeiT
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path /data/datasets/ImageNet1k --output_dir /data/yzy/bimask/deit_imagenet

cd DeiT
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path /data/datasets/ImageNet1k --output_dir /data/yzy/bimask/deit_imagenet  --eval

hetao new

cd ~

rsync -avP /data/lishen/yzy/ILSVRC2012_img_train.tar ./
mkdir imagenet
cd imagenet
mkdir train && tar -xvf ../ILSVRC2012_img_train.tar -C train && for x in `ls train/*tar`; do fn=train/`basename $x .tar`; mkdir $fn; tar -xvf $x -C $fn; rm -f $fn.tar; done

rsync -avP /data/lishen/yzy/ILSVRC2012_img_val.tar ./ 
mkdir -p imagenet/val
cd imagenet/
tar -xvf ../ILSVRC2012_img_val.tar -C val
cp /data/lishen/yzy/valprep.sh val/
cd imagenet/val/
bash valprep.sh

git clone git@github.com:fabfish/Bi-Mask.git


CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4  --use_env main.py --model vit_deit_small_patch16_224 --batch-size 256 --data-path /root/imagenet --output_dir /root/deit_imagenet --num_workers 0 --no-pin-mem



pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

/root/Bi-Mask/datasets

python cifar.py --arch resnet32_cifar10 --lr 0.1 --weight_decay 0.001 --data_path /root/Bi-Mask/datasets --label_smoothing 0.1 --num_epochs 300 --job_dir /root/resnet32_cifar10_test

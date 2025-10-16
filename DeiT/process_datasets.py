import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# 1. 定义路径
HF_DATASET_PATH = "/hefhspace/yzy/imagenet-1k"  # 您的 HF 格式的本地路径
OUTPUT_DIR = "/hefhspace/yzy/imagenet-1k_imagefolder" # 目标 ImageFolder 路径

# 2. 转换函数
def convert_to_imagefolder(hf_dataset_path, output_dir, split):
    print(f"Loading {split} split from HF format...")
    # 加载已缓存的 HF 数据集（此步骤应快速完成，因为它从缓存读取）
    hf_dataset = load_dataset(hf_dataset_path, split=split)

    # 创建输出目录
    split_dir = os.path.join(output_dir, split)
    os.makedirs(split_dir, exist_ok=True)

    print(f"Converting {split} split to ImageFolder structure...")
    
    # 遍历数据集
    for i, item in enumerate(tqdm(hf_dataset)):
        # 标签和图像
        image: Image.Image = item["image"]
        label = item["label"]  # 0-999 的整数标签

        # ImageNet 标准：使用 WordNet ID (nxxxxxxx) 作为文件夹名。
        # 注意：HF 缓存中的 'label' 是 0-999 的整数，您需要一个映射表
        # 将这些整数映射回 WordNet ID (nxxxxxxx) 字符串。
        # 假设我们暂时用整数作为文件夹名 (这对 ImageFolder 是可行的, 但非标准)
        class_name = str(label) # 临时使用整数标签作为文件夹名

        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # 保存图片，使用索引作为文件名
        filename = f"{i:08d}.jpeg" # 使用一个唯一的八位数字 ID
        output_path = os.path.join(class_dir, filename)
        
        # 保存为 JPEG 格式
        image.save(output_path, "JPEG")

    print(f"Conversion of {split} split completed in: {split_dir}")

# 3. 执行转换
# 警告：此过程需要大量磁盘 I/O，并且会占用双倍磁盘空间！
convert_to_imagefolder(HF_DATASET_PATH, OUTPUT_DIR, "train")
convert_to_imagefolder(HF_DATASET_PATH, OUTPUT_DIR, "validation")

print("All conversion finished. You can now use the path:", OUTPUT_DIR)
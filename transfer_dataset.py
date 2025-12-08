import os
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# 1. 创建保存目录
os.makedirs("datasets/coralscapes/train_img", exist_ok=True)
os.makedirs("datasets/coralscapes/train_label", exist_ok=True)

print("正在从 Hugging Face 下载数据...")
# 加载数据集 (会自动下载到本地缓存)
dataset = load_dataset("EPFL-ECEO/coralscapes")

print("正在将数据保存为图片文件...")

# 遍历训练集
# 注意：根据截图，列名是 'image' 和 'label'
for idx, item in enumerate(tqdm(dataset['train'])):
    # 获取图片和掩码对象 (PIL Image)
    image = item['image']
    label = item['label']
    
    # 构造文件名 (例如 0001.png)
    file_name = f"{idx:05d}.png"
    
    # 保存图片 (原图 RGB)
    image.save(os.path.join("datasets/coralscapes/train_img", file_name))
    
    # 保存标签 (掩码通常是单通道，如果是RGB掩码可能需要转换，这里直接保存)
    # Coralscapes 的 label 看起来已经是灰度或索引图了，直接保存即可
    label.save(os.path.join("datasets/coralscapes/train_label", file_name))

print("下载并转换完成！")
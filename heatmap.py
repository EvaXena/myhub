import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def visualize_feature_maps(image_path):
    # 1. 准备工作：加载模型
    # 修正警告：使用 weights 参数替代 pretrained
    weights = models.VGG16_Weights.IMAGENET1K_V1
    model = models.vgg16(weights=weights)
    
    # 提取模型中的特征提取层 (features)
    # vgg16.features 包含了所有的卷积层和池化层
    model_children = list(model.features.children())
    
    # 2. 预处理图片
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])
    
    # 确保读取图片
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"❌ 错误：找不到图片文件 {image_path}，请检查路径！")
        return

    image_tensor = preprocess(image)
    image_tensor = image_tensor.unsqueeze(0) # [1, 3, 224, 224]

    # 3. 将图片一层一层传进去，保存结果
    # ✅ 修正点：直接把初始 tensor 放进列表，而不是用 append 的返回值
    results = [image_tensor] 
    
    print(f"模型共有 {len(model_children)} 层，正在逐层推理...")
    
    for i, layer in enumerate(model_children):
        # 让图片数据流过这一层
        image_tensor = layer(image_tensor)
        results.append(image_tensor)
        
        # 为了防止内存爆炸，如果层数太多，就不存后面的了
        if i > 20: break 

    # 4. 画图
    # 选择我们要可视化的层索引
    # 注意：VGG16的第0层是卷积，第1层是ReLU... 建议看卷积层的输出
    visualize_layers = [2, 8, 14, 20] 
    
    for layer_num in visualize_layers:
        # 检查索引是否越界
        if layer_num >= len(results):
            print(f"跳过第 {layer_num} 层（超出范围）")
            continue

        plt.figure(figsize=(20, 10))
        
        # 取出这一层的输出，维度是 [1, Channel, H, W]
        # squeeze(0) 去掉 batch 维度 -> [Channel, H, W]
        layer_viz = results[layer_num].squeeze(0).data
        
        print(f"正在可视化第 {layer_num} 层输出，特征图形状: {layer_viz.shape}")
        
        # 只画前 16 个通道 (Channel)
        num_channels = min(16, layer_viz.size(0))
        for i in range(num_channels):
            plt.subplot(2, 8, i + 1)
            # 这里的 cmap='viridis' 也就是热力图配色
            plt.imshow(layer_viz[i], cmap='viridis') 
            plt.axis('off')
            plt.title(f"L{layer_num}-Ch{i}")
            
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # 请确保同级目录下有一张叫 coral_image.jpg 的图片
    # 或者把这里改成你实际图片的路径
    visualize_feature_maps('1.jpg')
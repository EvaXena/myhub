from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import cv2
import numpy as np
import torch

# 1. 加载模型
model = resnet50(pretrained=True)
target_layers = [model.layer4[-1]] # 通常选最后一个卷积层，因为它语义信息最丰富

# 2. 准备图片 (需要归一化到 0-1 之间)
rgb_img = cv2.imread("1.jpg")[:, :, ::-1] # BGR转RGB
rgb_img = np.float32(rgb_img) / 255
input_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1).unsqueeze(0) # 调整形状

# 3. 实例化 CAM 对象
cam = GradCAM(model=model, target_layers=target_layers)

# 4. 生成热力图
# targets=None 表示让它自动找概率最大的那个类别
grayscale_cam = cam(input_tensor=input_tensor, targets=None)

# 5. 将热力图叠加到原图上
grayscale_cam = grayscale_cam[0, :]
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

# 6. 显示
import matplotlib.pyplot as plt
plt.imshow(visualization)
plt.show()
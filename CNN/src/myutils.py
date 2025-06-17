import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms

# 重构误差计算函数
def reconstruction_error(input, output):
    """计算输入和输出之间的重构误差"""
    return torch.mean((input - output) ** 2, dim=[1, 2, 3])

# 图像变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 滑动窗口生成
def get_box(box_size=(128,128), box_rec=(16,16), start_lt=(0,0)):
    h, w = box_size[0], box_size[1]
    r, c = box_rec[0], box_rec[1]
    l, t = start_lt[0], start_lt[1]
    boxs = np.zeros((r*c, 4))
    for i in range(r) :
        for j in range(c) :
            boxs[i*c+j] = (l+j*w, t+i*h, l+(j+1)*w, t+(i+1)*h)
    return boxs.astype(np.int32)

# 预测函数（使用重构误差）
def predict(model, img, boxs):
    imgs = torch.zeros(boxs.shape[0], 3, 256, 256)
    for i, box in enumerate(boxs):
        im = img.crop(box).resize([256,256])
        imgs[i] = transform(im)
    
    imgs = imgs.cuda()
    with torch.no_grad():
        outputs = model(imgs)
        errors = reconstruction_error(imgs, outputs)
    
    return errors.cpu().numpy()
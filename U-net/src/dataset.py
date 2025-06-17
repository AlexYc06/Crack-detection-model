import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np

class myDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        Dataset.__init__(self)
        images_dir = os.path.join(root, 'images')
        images = os.listdir(images_dir)
        self.images = [os.path.join(images_dir, k) for k in images]
        self.images.sort()
        if train:
            masks_dir = os.path.join(root, 'masks')
            masks = os.listdir(masks_dir)
            self.masks = [os.path.join(masks_dir, k) for k in masks]
            self.masks.sort()

        self.transforms = transform
        self.train = train
        
    def __getitem__(self, index):
        image_path = self.images[index]
        
        image = Image.open(image_path).resize([512, 512])
        if self.transforms is not None:
            # 对图像应用完整transform（包括归一化）
            image = self.transforms(image)
        
        if self.train:
            mask_path = self.masks[index]
            mask = Image.open(mask_path).resize([512, 512])
            
            # 对mask单独处理：只转换为Tensor，不应用归一化
            mask = transforms.ToTensor()(mask)
            # 取平均并保持单通道
            mask = mask.mean(dim=0, keepdim=True)
            # 二值化处理：大于0.5的值设为1，其余设为0
            mask = (mask > 0.5).float()
                
            return image, mask
        return image
    
    def __len__(self):
        return len(self.images)
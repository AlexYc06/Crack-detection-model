import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class CrackDataset(Dataset):
    def __init__(self, root, transform=None):
        self.images_dir = os.path.join(root, 'images')
        
        # 获取所有图像文件
        self.image_paths = [
            os.path.join(self.images_dir, f) 
            for f in os.listdir(self.images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        
        print(f"找到 {len(self.image_paths)} 个图像文件")
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"在 {self.images_dir} 中未找到图像文件")
        
        self.transform = transform
    
    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB').resize([256, 256])
        
        if self.transform:
            img = self.transform(img)
        
        # 自编码器只需要输入图像（无标签）
        return img
    
    def __len__(self):
        return len(self.image_paths)
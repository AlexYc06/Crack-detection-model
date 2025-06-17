import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse  # 添加参数解析模块

from myutils import transform
from model import Unet

def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--model_path', type=str, default='./trained_models/unet_adam_bce_31.pkl',
                        help='模型权重文件路径')
    parser.add_argument('--input_dir', type=str, default='./data/test/images/',
                        help='输入图像目录')
    parser.add_argument('--output_dir', type=str, default='./data/test/pred_adam_bce/',
                        help='预测结果输出目录')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 获取输入图像列表
    img_list = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    img_list.sort()

    # 加载模型
    unet = Unet(3, 2).cuda()
    unet.load_state_dict(torch.load(args.model_path))
    unet.eval()  # 设置为评估模式

    print(f"使用模型: {args.model_path}")
    print(f"处理 {len(img_list)} 张图像...")
    
    # 处理每张图像
    for file in img_list:
        # 加载并预处理图像
        img = Image.open(file).resize([512, 512])
        img_tensor = transform(img).cuda().unsqueeze(0)
        
        # 模型预测
        with torch.no_grad():
            pred = unet(img_tensor)
        
        # 获取裂缝预测概率图
        # 修改1: 使用argmax代替argmin
        pred_mask = torch.argmax(pred, 1).float()
        
        # 修改2: 反转预测结果（黑底白线）
        # 原始: 裂缝=1(白), 背景=0(黑) → 反转: 裂缝=0(黑), 背景=1(白)
        # 但我们希望: 裂缝=1(白), 背景=0(黑) → 所以不需要反转，但需要调整输出
        # 直接使用裂缝通道作为预测结果
        
        # 获取裂缝通道的概率图
        crack_prob = pred[:, 1, :, :]  # 裂缝是第1个通道
        
        # 将概率图转换为二值图
        pred_binary = (crack_prob > 0.5).float()  # 阈值设为0.5
        
        # 转换为numpy数组并调整
        pred_np = pred_binary.squeeze().cpu().numpy()
        
        # 修改3: 反转颜色：裂缝=255(白), 背景=0(黑)
        # 因为当前裂缝=1, 背景=0 → 乘以255后裂缝=255(白), 背景=0(黑)
        pred_np = np.uint8(pred_np * 255)
        
        # 保存预测结果
        pred_img = Image.fromarray(pred_np)
        img_name = os.path.basename(file)
        output_path = os.path.join(args.output_dir, f"pred_{img_name}")
        pred_img.save(output_path, 'PNG')
        print(f"已保存: {output_path}")

    print("预测完成！所有结果已转换为黑底白线")

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
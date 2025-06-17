import os
import torch
import numpy as np
from PIL import Image
from model import CrackAutoencoder
from myutils import get_box, predict

def main():
    # 配置参数
    img_folder = './data/test/images/'
    output_dir = './data/test/cnn_pred/'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    model = CrackAutoencoder().cuda()
    model.load_state_dict(torch.load('./trained_models/best_autoencoder.pth'))
    model.eval()
    
    # 生成滑动窗口
    boxs_full = get_box(box_size=(128,128), box_rec=(16,16))
    boxs_center = get_box(box_size=(128,128), box_rec=(15,15), start_lt=(64,64))
    
    # 获取测试图像
    img_files = [os.path.join(img_folder, f) for f in os.listdir(img_folder) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for file_path in img_files:
        # 处理每张图像
        img = Image.open(file_path).convert('RGB').resize([2048,2048])
        
        # 计算重构误差
        errors_full = predict(model, img, boxs_full)
        errors_center = predict(model, img, boxs_center)
        
        # 合并误差图
        error_map = np.zeros((32, 32))
        
        # 填充完整区域误差
        for i, box in enumerate(boxs_full):
            x1, y1, x2, y2 = box
            grid_x = int(i % 16) * 2
            grid_y = int(i // 16) * 2
            error_map[grid_y:grid_y+2, grid_x:grid_x+2] = errors_full[i]
        
        # 填充中心区域误差
        for i, box in enumerate(boxs_center):
            x1, y1, x2, y2 = box
            grid_x = int(i % 15) * 2 + 1
            grid_y = int(i // 15) * 2 + 1
            error_map[grid_y:grid_y+2, grid_x:grid_x+2] += errors_center[i]
        
        # 归一化误差图 (0-255)
        error_map = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-7)
        error_map = (error_map * 255).astype(np.uint8)
        
        # 创建热力图
        heatmap = Image.fromarray(error_map).resize([512, 512], Image.BILINEAR)
        
        # 保存结果
        file_name = os.path.basename(file_path)
        save_path = os.path.join(output_dir, f"heatmap_{file_name}")
        heatmap.save(save_path)
        print(f"保存热力图: {save_path}")

if __name__ == '__main__':
    main()
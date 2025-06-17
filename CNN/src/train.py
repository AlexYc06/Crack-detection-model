import os
import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader
from dataset import CrackDataset
from model import CrackAutoencoder
from myutils import transform

def main():
    # 配置参数
    batch_size = 32
    num_epochs = 200
    learning_rate = 0.001
    torch.manual_seed(777)
    
    # 创建数据集
    train_path = './data/train'
    val_path = './data/val'
    
    train_dataset = CrackDataset(train_path, transform=transform)
    val_dataset = CrackDataset(val_path, transform=transform)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = CrackAutoencoder().cuda()
    criterion = nn.MSELoss()  # 使用均方误差损失
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 日志配置
    os.makedirs('./logs', exist_ok=True)
    logging.basicConfig(filename='./logs/autoencoder.log', level=logging.INFO)
    
    # 训练循环
    best_loss = float('inf')
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for images in train_loader:
            images = images.cuda()
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, images)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.cuda()
                outputs = model(images)
                loss = criterion(outputs, images)
                val_loss += loss.item() * images.size(0)
        
        # 计算平均损失
        train_loss /= len(train_dataset)
        val_loss /= len(val_dataset)
        
        # 日志记录
        log_msg = f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        print(log_msg)
        logging.info(log_msg)
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs('./trained_models', exist_ok=True)
            torch.save(model.state_dict(), './trained_models/best_autoencoder.pth')
            print(f"保存最佳模型，验证损失: {val_loss:.6f}")

if __name__ == '__main__':
    main()
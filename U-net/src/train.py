import logging
import copy
import time
import os
import datetime  # 导入datetime模块用于时间格式化
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from myutils import transform, cal_iou, onehot
from model import Unet
from dataset import myDataset

import lovasz_losses as L
from Dice_coeff_loss import dice_loss
from focalloss import FocalLoss

batch_size = 2
num_epochs = [41, 41, 41, 41, 41, 41, 41, 41]
num_workers = 0
lr = 0.0001

losslist = ['focal', 'bce', 'dice', 'lovasz']
optimlist = ['adam', 'sgd']
iflog = True

train_dataset = myDataset('./data/train', transform=transform)
val_dataset = myDataset('./data/val', transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_workers)
val_loader = DataLoader(dataset=val_dataset,
                            batch_size=1,
                            shuffle=False)
criterion = nn.BCELoss()
focallos = FocalLoss(gamma=2)

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    
    # 确保创建必要的目录
    os.makedirs('./trained_models_2', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    # 记录整个训练开始时间
    total_start_time = time.time()
    
    # 添加路径验证
    print(f"训练集路径: {os.path.abspath('./data/train')}")
    print(f"验证集路径: {os.path.abspath('./data/val')}")
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    epoidx = -1
    for los in losslist:
        for opt in optimlist:
            # 记录当前配置开始时间
            config_start_time = time.time()
            print(f"\n{'='*50}")
            print(f"开始训练: 损失函数={los}, 优化器={opt}")
            print(f"开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*50}")
            
            torch.manual_seed(77)
            torch.cuda.manual_seed(77)
            unet = Unet(3,2).cuda()
            history = []
            if 'adam' in opt :
                optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
            elif 'sgd' in opt:
                optimizer = torch.optim.SGD(unet.parameters(), lr=10*lr, momentum=0.9)

            logging.basicConfig(filename='./logs/logger_unet.log', level=logging.INFO)

            total_step = len(train_loader)
            epoidx += 1
            for epoch in range(num_epochs[epoidx]):
                # 记录当前epoch开始时间
                epoch_start_time = time.time()
                
                totalloss = 0
                for i, (images, masks) in enumerate(train_loader):
                    images = images.cuda()
                    masks = masks.cuda()
                    outputs = unet(images)
                    if 'bce' in los :
                        masks = onehot(masks)              
                        loss = criterion(outputs, masks)
                    elif 'dice' in los :
                        masks = onehot(masks)              
                        loss = dice_loss(outputs, masks)
                    elif 'lovasz' in los :
                        masks = onehot(masks)          
                        loss = L.lovasz_hinge(outputs, masks)
                    elif 'focal' in los :
                        loss = focallos(outputs, masks.long())

                    totalloss += loss*images.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if i+1 == total_step:
                        train_pa, train_mpa, train_miou, train_fwiou = \
                                            cal_iou(unet,train_dataset)
                        val_pa, val_mpa, val_miou, val_fwiou = \
                                            cal_iou(unet,val_dataset)
                        history.append([totalloss.item()/len(train_dataset), 
                                        train_pa, train_mpa, train_miou, train_fwiou,
                                        val_pa, val_mpa, val_miou, val_fwiou])
                    
                    if  i+1 == total_step and epoch%10==0 :
                        torch.save(unet.state_dict(), 
                                    './trained_models_2/unet_'+opt+'_'+los+'_'+str(epoch)+'.pkl')
                
                # 计算当前epoch耗时
                epoch_time = time.time() - epoch_start_time
                
                # 打印epoch信息
                print(f"Epoch [{epoch+1}/{num_epochs[epoidx]}] - "
                      f"训练耗时: {epoch_time:.2f}秒 | "
                      f"平均损失: {totalloss.item()/len(train_dataset):.4f} | "
                      f"验证mIoU: {val_miou:.4f} | "
                      f"验证PA: {val_pa:.4f}")
                
                # 保存历史记录
                history_np = np.array(history)
                np.save('./logs/unet_'+opt+'_'+los+'.npy', history_np)
            
            # 计算当前配置总耗时
            config_time = time.time() - config_start_time
            config_minutes = config_time / 60
            print(f"\n{'='*50}")
            print(f"完成训练: 损失函数={los}, 优化器={opt}")
            print(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"总训练时间: {config_minutes:.2f}分钟")
            print(f"{'='*50}\n")
    
    # 计算整个训练过程总耗时
    total_time = time.time() - total_start_time
    total_minutes = total_time / 60
    total_hours = total_minutes / 60
    
    print(f"\n{'='*50}")
    print(f"所有配置训练完成!")
    print(f"开始时间: {datetime.datetime.fromtimestamp(total_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总训练时间: {total_hours:.2f}小时 ({total_minutes:.2f}分钟)")
    print(f"{'='*50}")
'''
Author: HaoZhi
Date: 2022-08-19 15:05:43
LastEditors: HaoZhi
LastEditTime: 2022-10-08 10:05:12
Description: 
'''
import os
import sys
sys.path.append(os.path.dirname(__file__))

from torch.utils.data import DataLoader

from dataset import CustomDataSet, BalanceSampler, BalanceBatchSampler, CommonDataSet
from augmentation import train_aug, valid_aug, train_aug_v2, MixUp

def build_dataloader(data_folder, batch_size, img_size, mode):
    if mode == 'train':
        data_aug = train_aug(img_size= img_size)
        dataset = CustomDataSet(data_folder= data_folder, data_aug= data_aug)
        sampler = BalanceSampler(data_source= dataset)
        batch_sampler = BalanceBatchSampler(sampler= sampler, batch_size= batch_size, drop_last= False)
        dataloader = DataLoader(dataset= dataset, batch_sampler= batch_sampler)
    else:
        data_aug = valid_aug(img_size= img_size)
        dataset = CommonDataSet(data_folder= data_folder, data_aug= data_aug, mode ='valid')
        dataloader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False)
        
    return dataloader

def build_dataloader_v2(data_folder, batch_size, img_size, mode):
    if mode == 'train':
        data_aug = train_aug_v2(img_size= img_size)
        dataset = CustomDataSet(data_folder= data_folder, data_aug= data_aug)
        sampler = BalanceSampler(data_source= dataset)
        batch_sampler = BalanceBatchSampler(sampler= sampler, batch_size= batch_size, drop_last= False)
        dataloader = DataLoader(dataset= dataset, batch_sampler= batch_sampler)
    else:
        data_aug = valid_aug(img_size= img_size)
        dataset = CommonDataSet(data_folder= data_folder, data_aug= data_aug, mode ='valid')
        dataloader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False)
        
    return dataloader

def build_dataloader_mixup(data_folder, batch_size, img_size, mode, num_class):
    if mode == 'train':
        print('mixup')
        mixup_fn = MixUp(num_class= num_class)
        data_aug = train_aug(img_size= img_size)
        dataset = CustomDataSet(data_folder= data_folder, data_aug= data_aug)
        sampler = BalanceSampler(data_source= dataset)
        batch_sampler = BalanceBatchSampler(sampler= sampler, batch_size= batch_size, drop_last= False)
        dataloader = DataLoader(dataset= dataset, batch_sampler= batch_sampler, collate_fn= mixup_fn)
    else:
        data_aug = valid_aug(img_size= img_size)
        dataset = CommonDataSet(data_folder= data_folder, data_aug= data_aug, mode ='valid')
        dataloader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False)
        
    return dataloader

def build_dataloader_unbalance(data_folder, batch_size, img_size, mode):
    if mode == 'train':
        data_aug = train_aug(img_size= img_size)
        dataset = CommonDataSet(data_folder= data_folder, data_aug= data_aug, mode = 'train')
        dataloader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= True)
    else:
        data_aug = valid_aug(img_size= img_size)
        dataset = CommonDataSet(data_folder= data_folder, data_aug= data_aug, mode = 'valid')
        dataloader = DataLoader(dataset= dataset, batch_size= batch_size, shuffle= False)
        
    return dataloader

if __name__ == '__main__':
    data_loader = build_dataloader_mixup(r'D:\workspace\convnext\dataset\train', 3, 224, 'train', 3)

    for e in range(5):
        for idx, (img, label, path) in enumerate(data_loader):
            print(list(zip(label, path)))
            if idx == 3:
                break
        # for idx, (img, path) in enumerate(data_loader):
        #     print(path)
        #     if idx == 3:
        #         break
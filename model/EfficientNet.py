'''
Author: HaoZhi
Date: 2022-08-26 14:41:51
LastEditors: HaoZhi
LastEditTime: 2022-08-26 14:51:17
Description: 
'''
from statistics import mode
import torch
from timm.models.efficientnet import efficientnet_b1

def effnet(num_class):
    model = efficientnet_b1(num_classes = num_class)
    return model

if __name__ == '__main__':
    # x = torch.randn(1, 3, 224, 224)
    # model = effnet(3)
    # out = model(x)
    # print(out.shape)
    from torchsummary import summary
    model = effnet(3)
    summary(model, (3, 224, 224))
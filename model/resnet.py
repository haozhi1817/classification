'''
Author: HaoZhi
Date: 2022-08-08 17:53:59
LastEditors: HaoZhi
LastEditTime: 2022-08-19 14:16:52
Description: 
'''
import torch
from timm.models.resnet import resnet50

def resnet(num_class):
    model = resnet50(num_classes = num_class)
    return model

if __name__ == '__main__':
    #from torchsummary import summary
    model = resnet(3)
    inputs = torch.zeros(size = (4, 3, 1024, 1024))
    logits = model(inputs)
    print(logits.shape)
    #summary(model, (3, 224, 224))

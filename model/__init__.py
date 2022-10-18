'''
Author: HaoZhi
Date: 2022-08-19 10:43:26
LastEditors: HaoZhi
LastEditTime: 2022-10-11 09:28:45
Description: 
'''
import os
import sys
from importlib import import_module

import  torch

sys.path.append(os.path.dirname(__file__))

def build_backbone(model_name, img_size, num_class):
    print('backbone: ', model_name)
    if model_name == 'resnet50':
        backbone = import_module('resnet')
        backbone = getattr(backbone, 'resnet')
        backbone = backbone(num_class = num_class)
    elif model_name == 'convnext':
        backbone = import_module('convnext')
        backbone = getattr(backbone, 'convnext_tiny')
        backbone = backbone(num_class = num_class)
    elif model_name == 'swin_t':
        backbone = import_module('swin_transformrt_official')
        backbone = getattr(backbone, 'SwinTransformer')
        backbone = backbone(img_size = img_size, num_classes = num_class)
    elif model_name == 'swin_t_p8_w7':
        backbone = import_module('swin_transformrt_official')
        backbone = getattr(backbone, 'SwinTransformer')
        backbone = backbone(img_size = img_size, num_classes = num_class, patch_size = 8)
    elif model_name == 'swin_t_p4_w14':
        backbone = import_module('swin_transformrt_official')
        backbone = getattr(backbone, 'SwinTransformer')
        backbone = backbone(img_size = img_size, num_classes = num_class, window_size=14)
    elif model_name == 'alternet':
        backbone = import_module('alternet')
        backbone = getattr(backbone, 'alternet_50')
        backbone = backbone(num_class = num_class)
    elif model_name == 'efficient_b1':
        backbone = import_module('EfficientNet')
        backbone = getattr(backbone, 'effnet')
        backbone = backbone(num_class = num_class)
    elif model_name == 'coatnet':
        backbone = import_module('coatnet')
        backbone = getattr(backbone, 'coatnet_2')
        backbone = backbone(img_size = img_size, num_class = num_class)
    elif model_name == 'maxvit':
        backbone = import_module('maxvit')
        backbone = getattr(backbone, 'maxvit_s')
        backbone = backbone(in_chans =3, num_class = num_class)
    else:
        raise NotImplementedError
    return backbone

def build_loss(loss_name):
    print('loss_name: ', loss_name)
    loss_module = import_module('loss')
    if loss_name == 'ce_loss':
        loss_op = getattr(loss_module, 'CE_loss')
    elif loss_name == 'ce_loss_smooth':
        loss_op = getattr(loss_module, 'CE_loss_Smooth')
    elif loss_name == 'focal_loss':
        loss_op = getattr(loss_module, 'FC_loss')
    else:
        raise NotImplementedError
    return loss_op()


if __name__ =='__main__':
    backbone = build_backbone('alternet', 448, 3)
    inputs = torch.zeros((4, 3, 448, 448))
    outputs = backbone(inputs)
    print(outputs)
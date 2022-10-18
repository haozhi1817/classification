'''
Author: HaoZhi
Date: 2022-08-19 14:49:02
LastEditors: HaoZhi
LastEditTime: 2022-10-11 11:15:01
Description: 
'''
import torch
import torch.nn as nn

class CE_loss(nn.Module):
    def __init__(self) -> None:
        super(CE_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss

class CE_loss_Smooth(nn.Module):
    def __init__(self) -> None:
        super(CE_loss_Smooth, self).__init__()
        self.loss = nn.CrossEntropyLoss(label_smoothing= 0.1)

    def forward(self, logits, labels):
        loss = self.loss(logits, labels)
        return loss

class FC_loss(nn.Module):
    def __init__(self) -> None:
        super(FC_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction= 'none')

    def forward(self, logits, labels):
        probs = torch.nn.functional.softmax(logits, dim = 1)
        probs = torch.gather(probs, 1, labels.unsqueeze(1)).squeeze(1)
        weight = (1 - probs) ** 2
        ce_loss = self.loss(logits, labels)
        loss = weight * ce_loss
        loss = 20 * torch.mean(loss)
        return loss

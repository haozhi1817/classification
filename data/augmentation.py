"""
Author: HaoZhi
Date: 2022-08-09 11:28:31
LastEditors: HaoZhi
LastEditTime: 2022-08-09 17:31:12
Description: 
"""
import math

import torch
import torch.nn as nn
from torchvision import transforms


class MixUp(nn.Module):
    def __init__(self, num_class, p=0.5, alpha=1.0, inplace=False) -> None:
        super().__init__()
        self.num_class = num_class
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch):
        imgs = torch.stack([data[0] for data in batch], axis=0)
        labels = torch.stack([torch.tensor(data[1]) for data in batch], axis=0)
        pathes = [data[2] for data in batch]
        #print("debug: ", imgs.shape, labels.shape)
        if not self.inplace:
            imgs = imgs.clone()
            labels = labels.clone()

        if labels.ndim == 1:
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_class).float()

        if torch.rand(1).item() >= self.p:
            return imgs, labels, pathes

        imgs_shift = torch.flip(imgs, dims=[0])
        labels_shift = torch.flip(labels, dims=[0])

        lamb = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        imgs_shift.mul_(1.0 - lamb)
        imgs.mul_(lamb).add_(imgs_shift)

        labels_shift.mul_(1.0 - lamb)
        labels.mul_(lamb).add_(labels_shift)
        return imgs, labels, pathes


class CutMix(nn.Module):
    def __init__(self, num_class, p, alpha, inplace) -> None:
        super().__init__()
        self.num_class = num_class
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(self, batch):
        imgs, labels, pathes = batch
        if not self.inplace:
            imgs = imgs.clone()
            labels = labels.clone()

        if labels.ndim == 1:
            labels = torch.nn.functional.one_hot(labels, num_classes=self.num_class)

        if torch.rand(1).item() >= self.p:
            return imgs, labels, pathes

        imgs_shift = imgs[::-1]
        labels_shift = labels[::-1]

        lamb = float(torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])
        W, H = imgs.shape[2:]
        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lamb)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        imgs[:, :, y1:y2, x1:x2] = imgs_shift[:, :, y1:y2, x1:x2]
        lamb = float(1 - (x2 - x1) * (y2 - y1) / (W * H))

        labels_shift.mul_(lamb).add_(labels_shift)
        return imgs, labels, pathes


def train_aug(img_size=512):
    print("augmode: TrivAug")
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.TrivialAugmentWide(
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            # transforms.ToTensor()
            transforms.ConvertImageDtype(torch.float),
        ]
    )
    return transform


def train_aug_v2(img_size=512):
    print("augmode: RandAug")
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                ratio=(3.0 / 4.0, 4.0 / 3.0),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.ToTensor()
            transforms.ConvertImageDtype(torch.float),
        ]
    )
    return transform


def valid_aug(img_size=512):
    transform = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ConvertImageDtype(torch.float),
        ]
    )
    return transform

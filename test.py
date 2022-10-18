"""
Author: HaoZhi
Date: 2022-08-08 15:21:39
LastEditors: HaoZhi
LastEditTime: 2022-08-08 15:23:21
Description: 
"""
import os
import glob
import random
from random import shuffle
from itertools import chain

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F


class Data(Dataset):
    def __init__(self):
        self.img = torch.cat([torch.ones(2, 2) for i in range(100)], dim=0)
        self.num_classes = 2
        self.label = torch.tensor(
            [random.randint(0, self.num_classes - 1) for i in range(100)]
        )

    def __getitem__(self, index):
        return self.img[index], self.label[index]

    def __len__(self):
        return len(self.label)


class CustomSampler(Sampler):
    def __init__(self, data):
        self.data = data

    def __iter__(self):
        indices = []
        for n in range(self.data.num_classes):
            index = torch.where(self.data.label == n)[0]
            indices.append(index)
        indices = torch.cat(indices, dim=0)
        return iter(indices)

    def __len__(self):
        return len(self.data)


class CustomBatchSampler:
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        i = 0
        sampler_list = list(self.sampler)
        for idx in sampler_list:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []

            if (
                i < len(sampler_list) - 1
                and self.sampler.data.label[idx]
                != self.sampler.data.label[sampler_list[i + 1]]
            ):
                if len(batch) > 0 and not self.drop_last:
                    yield batch
                    batch = []
                else:
                    batch = []
            i += 1
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


# def einsum1(feature, heads, dims):
#     features = torch.stack(feature.flatten(2).chunk(dims, axis = 1), 1)
#     features.permute_(0, 1, 3, 2)
#     return features

# def einsum2(features, x):
#     features = features.permute(0, 1, 3, 2)
#     features = features.flatten(1, 2)
#     features = torch.stack(features.chunk(x, -1), -2)
#     return features


if __name__ == "__main__":
    d = Data()
    s = CustomSampler(d)
    bs = CustomBatchSampler(s, 8, False)
    dl = DataLoader(d, batch_sampler=bs)
    for img, label in dl:
        print(label)

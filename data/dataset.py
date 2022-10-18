import os
import glob
from random import shuffle
from itertools import chain

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader

from augmentation import train_aug, valid_aug

class_dict = {'without_mask': 0, 'with_mask': 1, 'mask_weared_incorrect': 2}

class CustomDataSet(Dataset):
    def __init__(self, data_folder, data_aug) -> None:
        super(CustomDataSet, self).__init__()
        self.wo_mask_pathes = glob.glob(os.path.join(data_folder, 'without_mask', '*'))
        self.w_mask_pathes = glob.glob(os.path.join(data_folder, 'with_mask', '*'))
        self.ic_mask_pathes = glob.glob(os.path.join(data_folder, 'mask_weared_incorrect', '*'))
        self.num_wo_mask = len(self.wo_mask_pathes)
        self.num_w_mask = len(self.w_mask_pathes)
        self.num_ic_mask = len(self.ic_mask_pathes)
        self.total_files = self.wo_mask_pathes + self.w_mask_pathes + self.ic_mask_pathes
        self.wo_idx = list(range(0, self.num_wo_mask))
        self.w_idx = list(range(self.num_wo_mask, self.num_wo_mask + self.num_w_mask))
        self.ic_idx = list(range(self.num_wo_mask + self.num_w_mask, self.num_wo_mask + self.num_w_mask + self.num_ic_mask))
        self.data_aug = data_aug
    
    def __getitem__(self, index):
        file = self.total_files[index]
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        #print('img info: ', img.size, img.format, img.mode, file)
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        #print(img.shape, img.dtype)
        img = self.data_aug(img)
        # if self.data_aug:
        #     pass
        # else:
        #     resize = transforms.Resize(size = (512, 512), interpolation= F.InterpolationMode.BILINEAR)
        #     img = resize(img)
        label = class_dict[os.path.split(file)[0].split(os.sep)[-1]]
        return img, label, file

    def __len__(self):
        return min(self.num_wo_mask, self.num_w_mask, self.num_ic_mask)

class BalanceSampler(Sampler):
    def __init__(self, data_source) -> None:
        self.data_source = data_source
        self.wo_idx = self.data_source.wo_idx
        self.w_idx = self.data_source.w_idx
        self.ic_idx = self.data_source.ic_idx
        self.num_sample = len(self.data_source)

    def __iter__(self):
        shuffle(self.wo_idx)
        shuffle(self.w_idx)
        shuffle(self.ic_idx)
        total_idx = list(zip(self.wo_idx[:self.num_sample], self.w_idx[:self.num_sample], self.ic_idx[:self.num_sample]))
        return iter(total_idx)

    def __len__(self):
        return self.num_sample


class BalanceBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, drop_last):
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
    def __iter__(self):
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    batch = list(chain(*batch))
                    shuffle(batch)
                    yield batch
                except StopIteration:
                    break
            
        else:
            batch = [0] * self.batch_size * 3
            idx_in_batch = 0
            for (idx1, idx2, idx3) in self.sampler:
                batch[idx_in_batch] = idx1
                idx_in_batch += 1
                batch[idx_in_batch] = idx2
                idx_in_batch += 1
                batch[idx_in_batch] = idx3
                idx_in_batch += 1
                if idx_in_batch == self.batch_size * 3:
                    yield batch
                    idx_in_batch = 0
                    batch = [0] * self.batch_size * 3
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]


    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size -1) // self.batch_size


class CommonDataSet(Dataset):
    def __init__(self, data_folder, data_aug, mode) -> None:
        super(CommonDataSet, self).__init__()
        if mode == 'train':
            self.data_pathes = glob.glob(os.path.join(data_folder, '*/*'))
        else:
            self.data_pathes = glob.glob(os.path.join(data_folder, '*'))
        self.num_data = len(self.data_pathes)
        self.data_aug = data_aug
        self.mode = mode
    
    def __getitem__(self, index):
        file = self.data_pathes[index]
        img = Image.open(file)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        #print('img info: ', img.size, img.format, img.mode, file)
        img = np.array(img)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img)
        #print('ori: ', img.shape, img.dtype)
        img = self.data_aug(img)
        #print('aug: ', img.shape, img.dtype)
        # if self.data_aug:
        #     pass
        # else:
        #     resize = transforms.Resize(size = (512, 512), interpolation= F.InterpolationMode.BILINEAR)
        #     img = resize(img)
        if self.mode == 'train':
            label = class_dict[os.path.split(file)[0].split(os.sep)[-1]]
            return img, label, file
        else:
            return img,file

    def __len__(self):
        return self.num_data




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader
    dataset = CustomDataSet(data_folder= r'D:\workspace\convnext\dataset\train', data_aug= False)
    sampler = BalanceSampler(data_source= dataset)
    batch_sampler = BalanceBatchSampler(sampler= sampler, batch_size= 3, drop_last= False)
    trainloader = DataLoader(dataset= dataset, batch_sampler= batch_sampler)
    for (imgs, labels, pathes) in trainloader:
        print(imgs.shape)
        for (img, label, path) in zip(imgs, labels, pathes):
            img = img.numpy().astype('uint8')
            img = np.transpose(img, (1, 2, 0))
            target_folder = os.path.join(r'D:\workspace\convnext\debug', str(label.item()))
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)
            save_path = os.path.join(target_folder, path.split(os.sep)[-1])
            print(save_path, img.shape)
            plt.imsave(save_path, img)
    # for i in trainloader:
    #     #print(i)
    #     print(i[0].shape)
    #     break
    
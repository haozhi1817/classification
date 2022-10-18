"""
Author: HaoZhi
Date: 2022-08-12 16:17:52
LastEditors: HaoZhi
LastEditTime: 2022-08-12 16:41:51
Description: 
"""
import torch
import torch.nn as nn


class Mlp(nn.Module):
    def __init__(
        self, dim_in, dim_hidden=None, dim_out=None, act=nn.GELU, drop=0.0
    ) -> None:
        super().__init__()
        dim_out = dim_out or dim_in
        dim_hidden = dim_hidden or dim_in
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.act = act()
        self.fc2 = nn.Linear(dim_hidden, dim_out)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self, img_size, patch_size, in_chans, embed_dim=96, norm_layer=None
    ) -> None:
        super().__init__()
        self.img_size = (img_size, img_size)
        self.patch_size = (patch_size, patch_size)
        self.patch_resolution = [
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        ]
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # different from vit
        if self.norm is not None:
            x = self.norm(x)
        return x

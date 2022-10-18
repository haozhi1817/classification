"""
Author: HaoZhi
Date: 2022-08-10 14:00:14
LastEditors: HaoZhi
LastEditTime: 2022-08-10 14:09:00
Description: 
"""
from functools import partial
from turtle import forward

import torch
import numpy as np
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange


class Stem(nn.Module):
    def __init__(self, dim_in, dim_out, pool=True) -> None:
        super().__init__()
        self.layer0 = []

        if pool:
            self.layer0.append(
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                    groups=1,
                    bias=False,
                )
            )
            self.layer0.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        else:
            self.layer0.append(
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_out,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    bias=False,
                )
            )

        self.layer0 = nn.Sequential(*self.layer0)

    def forward(self, x):
        x = self.layer0(x)
        return x


class BNGAPBlock(nn.Module):
    def __init__(self, in_features, num_classes, **kwargs) -> None:
        super(BNGAPBlock, self).__init__()

        self.bn = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(in_features, num_classes, bias=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = x.view(x.size()[0], -1)
        x = self.dense(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        width_per_group=64,
        sd=0,
        **block_kwargs
    ) -> None:
        super(BasicBlock, self).__init__()

        if groups != 1 or width_per_group != 64:
            raise ValueError("BasicBlock only support groups =1 and base_width = 64")
        width = int(channels * (width_per_group / 64.0)) ** groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(
                nn.Conv2d(
                    in_channels,
                    channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(
            in_channels, width, stride=stride, bias=False, kernel_size=3, padding=1
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                width,
                channels * self.expansion,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sd(x) + skip
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels,
        channels,
        stride=1,
        groups=1,
        width_per_group=64,
        sd=0.0,
        **block_kwargs
    ) -> None:
        super(Bottleneck, self).__init__()
        width = int(channels * (width_per_group / 64.0)) * groups

        self.shortcut = []
        if stride != 1 or in_channels != channels * self.expansion:
            self.shortcut.append(
                nn.Conv2d(
                    in_channels,
                    channels * self.expansion,
                    kernel_size=1,
                    bias=False,
                    stride=stride,
                )
            )
        self.shortcut = nn.Sequential(*self.shortcut)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                width,
                width,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=groups,
                bias=False,
            ),
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                width,
                channels * self.expansion,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        )

        self.sd = DropPath(sd) if sd > 0 else nn.Identity()

    def forward(self, x):
        if len(self.shortcut) > 0:
            x = self.bn(x)
            x = self.relu(x)
            skip = self.shortcut(x)
        else:
            skip = self.shortcut(x)
            x = self.bn(x)
            x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.sd(x) + skip
        # print('bottleblock: ', x.shape)
        return x


class Attention2d(nn.Module):
    """
    输入为[B,C,W,H], 这里将WH视为字数, C视为每一个字的特征, 假设C = 128, 将其利用8头, 每头为12维度的attnblock运算后, 得到[B, 96, W, H]
    计算公式简化为： y = f((q(x)k(x).T)*v(x)), 其中共有4个变换函数, 函数有卷积实现, 因此只与卷积层输入特诊维度也就是每一个字的特征长度相关, 
    与字数无关, 换句话说可以在不同字数的数据上做迁移
    Parameters
    ----------
    nn : _type_
        _description_
    """

    def __init__(
        self, dim_in, dim_out=None, heads=8, dim_head=64, dropout=0.0, k=1
    ) -> None:
        super().__init__()
        self.heads = heads
        self.scale = dim_head**-0.5

        self.inner_dim = dim_head * heads
        dim_out = dim_in if dim_out is None else dim_out
        self.to_q = nn.Conv2d(dim_in, self.inner_dim * 1, 1, bias=False)
        self.to_kv = nn.Conv2d(dim_in, self.inner_dim * 2, k, stride=k, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(self.inner_dim, dim_out, 1),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        )

    def forward(self, x, mask=None):
        b, n, _, y = x.shape
        qkv = (self.to_q(x), *self.to_kv(x).chunk(2, dim=1))
        q, k, v = map(
            lambda t: rearrange(t, "b (h d) x y -> b h (x y) d", h=self.heads), qkv
        )  # (B, 8, wh, dims)
        dots = (
            torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        )  # (B, 8, wh, wh)
        dots = dots + mask if mask is not None else dots  # (B, 8, wh, wh)
        attn = dots.softmax(dim=-1)  # (B, 8, wh, wh)
        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)  # (B, 8, wh, dims)
        # ('attn: ', out.shape)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", y=y)  # (B, 8 * dims, w, h)
        out = self.to_out(out)
        return out, attn


class LocalAttention(nn.Module):
    '''
    将特征图再次patchify后做attention, 与传统的vit中不太一致的是v 传统vit中patchify以后,每一个字内部信息被压缩到特征维度上,这里则放到batch维度上, 接近传统意义上的local
    且其中这个mask实在不太理解。

    Parameters
    ----------
    nn : _type_
        _description_
    '''
    def __init__(
        self,
        dim_in,
        dim_out=None,
        window_size=7,
        k=1,
        heads=8,
        dim_head=32,
        dropout=0.0,
    ) -> None:
        super().__init__()
        self.attn = Attention2d(
            dim_in=dim_in,
            dim_out=dim_out,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
            k=k,
        )
        self.window_size = window_size
        self.rel_index = (self.rel_distance(window_size) + window_size - 1).long()
        self.pos_embedding = nn.Parameter(
            torch.randn(2 * window_size - 1, 2 * window_size - 1) * 0.02
        )
        # ('pos_embed: ', self.pos_embedding.dtype, self.pos_embedding.shape)

    @staticmethod
    def rel_distance(window_size):
        i = torch.tensor(
            np.array([[x, y] for x in range(window_size) for y in range(window_size)])
        )
        d = i[None, :, :] - i[:, None, :]
        # ('debug: ', d.dtype, d.shape)
        return d

    def forward(self, x, mask=None):
        b, c, h, w = x.shape
        p = self.window_size
        n1 = h // p
        n2 = w // p
        mask = torch.zeros(p**2, p**2, device=x.device) if mask is None else mask
        mask = (
            mask + self.pos_embedding[self.rel_index[:, :, 0], self.rel_index[:, :, 1]]
        )
        x = rearrange(
            x, "b c (n1 p1) (n2 p2) -> (b n1 n2) c p1 p2", p1=p, p2=p
        )  # (B, C, W , H) - > (B*n1 *n2, C, p, p)
        x, attn = self.attn(x, mask)
        x = rearrange(
            x, "(b n1 n2) c p1 p2 -> b c (n1 p1) (n2 p2)", n1=n1, n2=n2, p1=p, p2=p
        )
        return x, attn


class AttentionBlock(nn.Module):
    expansion = 4

    def __init__(
        self,
        dim_in,
        dim_out,
        heads=8,
        dim_head=64,
        dropout=0.0,
        sd=0.0,
        stride=1,
        window_size=7,
        k=1,
        norm=nn.BatchNorm2d,
        activation=nn.GELU,
        **blcok_kwargs
    ) -> None:
        super().__init__()
        dim_out = dim_in if dim_out is None else dim_out
        attn = partial(LocalAttention, window_size=window_size, k=k)
        width = dim_in // self.expansion

        self.short_cut = []
        if stride != 1 or dim_in != dim_out * self.expansion:
            self.short_cut.append(
                nn.Conv2d(
                    dim_in,
                    dim_out * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                )
            )
        self.short_cut = nn.Sequential(*self.short_cut)
        self.norm1 = norm(dim_in)
        self.relu = activation()

        self.conv = nn.Conv2d(dim_in, width, kernel_size=1, bias=False)
        self.norm2 = norm(width)
        self.attn = attn(
            width,
            dim_out * self.expansion,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )
        self.sd = DropPath(sd) if sd > 0.0 else nn.Identity()

    def forward(self, x):
        if len(self.short_cut) > 0:
            x = self.norm1(x)
            x = self.relu(x)
            skip = self.short_cut(x)
        else:
            skip = self.short_cut(x)
            x = self.norm1(x)
            x = self.relu(x)

        x = self.conv(x)
        x = self.norm2(x)
        # ('attentionblock: ', x.shape)
        x, attn = self.attn(x)
        x = self.sd(x) + skip
        return x


class AttentionBasicBlock(AttentionBlock):
    expansion = 1


class AlterNet(nn.Module):
    def __init__(
        self,
        block1,
        block2,
        num_blocks,
        num_blocks2,
        heads,
        cblock=BNGAPBlock,
        sd=0.0,
        num_classes=3,
        stem=Stem,
        name="alternet",
        **block_kwargs
    ) -> None:
        super().__init__()
        self.name = name
        idxs = [
            [j for j in range(sum(num_blocks[:i]), sum(num_blocks[: i + 1]))]
            for i in range(len(num_blocks))
        ]
        sds = [[sd * j / (sum(num_blocks) - 1) for j in js] for js in idxs]

        self.layer0 = stem(3, 64)
        self.layer1 = self._make_layer(
            block1,
            block2,
            64,
            64,
            num_blocks[0],
            num_blocks2[0],
            stride=1,
            heads=heads[0],
            sds=sds[0],
            **block_kwargs
        )
        self.layer2 = self._make_layer(
            block1,
            block2,
            64 * block2.expansion,
            128,
            num_blocks[1],
            num_blocks2[1],
            stride=2,
            heads=heads[1],
            sds=sds[1],
            **block_kwargs
        )
        self.layer3 = self._make_layer(
            block1,
            block2,
            128 * block2.expansion,
            256,
            num_blocks[2],
            num_blocks2[2],
            stride=2,
            heads=heads[2],
            sds=sds[2],
            **block_kwargs
        )
        self.layer4 = self._make_layer(
            block1,
            block2,
            256 * block2.expansion,
            512,
            num_blocks[3],
            num_blocks2[3],
            stride=2,
            heads=heads[3],
            sds=sds[3],
            **block_kwargs
        )
        self.classifier = []
        self.classifier.append(
            cblock(512 * block2.expansion, num_classes, **block_kwargs)
        )
        self.classifier = nn.Sequential(*self.classifier)

    @staticmethod
    def _make_layer(
        block1,
        block2,
        in_channels,
        out_channels,
        num_block1,
        num_block2,
        stride,
        heads,
        sds,
        **block_kwargs
    ):
        alt_seq = [False] * (num_block1 - num_block2 * 2) + [False, True] * num_block2
        stride_seq = [stride] + [1] * (num_block1 - 1)
        seq, channels = [], in_channels

        for alt, stride, sd in zip(alt_seq, stride_seq, sds):
            block = block1 if not alt else block2
            seq.append(
                block(
                    channels,
                    out_channels,
                    stride=stride,
                    sd=sd,
                    heads=heads,
                    **block_kwargs
                )
            )
            channels = out_channels * block.expansion

        return nn.Sequential(*seq)

    def forward(self, x):
        # print('input: ', x.shape)
        x = self.layer0(x)
        # print('c0: ', x.shape)
        x = self.layer1(x)
        # print('c1: ', x.shape)
        x = self.layer2(x)
        # print('c2: ', x.shape)
        x = self.layer3(x)
        # print('c3: ', x.shape)
        x = self.layer4(x)
        # print('c4: ', x.shape)
        x = self.classifier(x)
        return x


def alternet_18(num_class=3, stem=True, name="alternet_18", **block_kwargs):
    return AlterNet(
        BasicBlock,
        AttentionBasicBlock,
        stem=partial(Stem, pool=stem),
        num_blocks=(2, 2, 2, 2),
        num_blocks2=(0, 1, 1, 1),
        heads=(3, 6, 12, 24),
        num_classes=num_class,
        name=name,
        **block_kwargs
    )


def alternet_34(num_class=3, stem=True, name="alternet_34", **block_kwargs):
    return AlterNet(
        BasicBlock,
        AttentionBasicBlock,
        stem=partial(Stem, pool=stem),
        num_blocks=(3, 4, 6, 4),
        num_blocks2=(0, 1, 3, 2),
        heads=(3, 6, 12, 24),
        num_classes=num_class,
        name=name,
        **block_kwargs
    )


def alternet_50(num_class=3, stem=True, name="alternet_50", **block_kwargs):
    return AlterNet(
        Bottleneck,
        AttentionBlock,
        stem=partial(Stem, pool=stem),
        num_blocks=(3, 4, 6, 4),
        num_blocks2=(0, 1, 3, 2),
        heads=(3, 6, 12, 24),
        num_classes=num_class,
        name=name,
        **block_kwargs
    )


def alternet_101(num_class=3, stem=True, name="alternet_101", **block_kwargs):
    return AlterNet(
        Bottleneck,
        AttentionBlock,
        stem=partial(Stem, pool=stem),
        num_blocks=(3, 4, 23, 4),
        num_blocks2=(0, 1, 3, 2),
        heads=(3, 6, 12, 24),
        num_classes=num_class,
        name=name,
        **block_kwargs
    )


def alternet_152(num_class=3, stem=True, name="alternet_152", **block_kwargs):
    return AlterNet(
        Bottleneck,
        AttentionBlock,
        stem=partial(Stem, pool=stem),
        num_blocks=(3, 8, 36, 4),
        num_blocks2=(0, 1, 3, 2),
        heads=(3, 6, 12, 24),
        num_classes=num_class,
        name=name,
        **block_kwargs
    )


if __name__ == "__main__":
    from torchsummary import summary

    inputs = torch.zeros(16, 3, 224, 224)
    model = alternet_50()
    output = model(inputs)
    print(output.shape)
    summary(model, (3, 224, 224))

"""
Author: HaoZhi
Date: 2022-08-30 11:26:27
LastEditors: HaoZhi
LastEditTime: 2022-08-30 14:29:35
Description: 
"""
import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_


def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def grid_partition(x, grid_size):
    B, H, W, C = x.shape
    assert (
        H % grid_size == 0 and W % grid_size == 0
    ), f"Feature map size{(H, W)} not divisible by grid size {(grid_size)}"
    x = x.view(B, grid_size, H // grid_size, grid_size, W // grid_size, C)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size, grid_size, C)
    return windows


def grid_reverse(grids, grid_size, H, W):
    B = int(grids.shape[0] / (H * W / grid_size / grid_size))
    x = grids.view(B, H // grid_size, W // grid_size, grid_size, grid_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
    return x


class SE(nn.Module):
    def __init__(self, in_chans, se_filters, out_chans) -> None:
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_chans, se_filters, kernel_size=1, stride=1, bias=True),
            nn.GELU(),
            nn.Conv2d(se_filters, out_chans, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x):
        y = torch.sigmoid(self.se(x))
        x = y * x
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_chans,
        out_chans,
        expansion_rate=4,
        downsample=True,
        se_ratio=0.25,
        drop_path=0.0,
    ) -> None:
        super().__init__()

        stride = 2 if downsample else 1
        hidden_chans = int(in_chans * expansion_rate)

        self.residual_pre_norm = nn.BatchNorm2d(in_chans)

        if expansion_rate == 1:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(
                    in_chans,
                    hidden_chans,
                    kernel_size=3,
                    stride=stride,
                    bias=False,
                    groups=in_chans,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_chans),
                nn.GELU(),
            )
        else:
            self.residual_conv = nn.Sequential(
                nn.Conv2d(in_chans, hidden_chans, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_chans),
                nn.GELU(),
                nn.Conv2d(
                    hidden_chans,
                    hidden_chans,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                    groups=hidden_chans,
                ),
                nn.BatchNorm2d(hidden_chans),
                nn.GELU(),
            )

        self.se = (
            SE(
                in_chans=hidden_chans,
                se_filters=int(hidden_chans * se_ratio),
                out_chans=hidden_chans,
            )
            if se_ratio
            else nn.Identity()
        )

        self.shrink_conv = nn.Conv2d(
            hidden_chans, out_chans, kernel_size=1, stride=1, bias=True
        )

        self.downsample = (
            nn.AvgPool2d(kernel_size=2, stride=2) if downsample else nn.Identity()
        )
        self.shortcut_conv = (
            nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=1, bias=True)
            if expansion_rate > 1
            else nn.Identity()
        )

        self.drop_path = DropPath(drop_path) if drop_path else nn.Identity()

    def forward(self, x):
        residual = self.residual_branch(x)
        shortcut = self.shortcut_branch(x)
        output = shortcut + self.drop_path(residual)
        return output

    def residual_branch(self, x):
        x = self.residual_pre_norm(x)
        x = self.residual_conv(x)
        x = self.se(x)
        x = self.shrink_conv(x)
        return x

    def shortcut_branch(self, x):
        x = self.downsample(x)
        x = self.shortcut_conv(x)
        return x


class LoclaAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_size,
        head_dim,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        window_size = (window_size, window_size)
        self.window_size = window_size  # Wh, Ww
        self.num_heads = dim // head_dim
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1), self.num_heads
            )
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.ffn = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            act_layer(),
            nn.Dropout(drop),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = self.ffn(x)
        return x


class TransFormer(nn.Module):
    def __init__(
        self,
        in_chans=64,
        head_dim=32,
        partition_size=7,
        mlp_ratio=4.0,
        mode="block",
        drop_rate=0.0,
        drop_path_rate=0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(in_chans)
        self.attn = LoclaAttention(
            dim=in_chans,
            head_dim=head_dim,
            window_size=partition_size,
            attn_drop=drop_rate,
            proj_drop=drop_rate,
        )
        self.ffn = FFN(
            in_features=in_chans,
            hidden_features=int(in_chans * mlp_ratio),
            drop=drop_rate,
        )
        self.partition_size = partition_size
        self.mode = mode
        self.norm2 = nn.LayerNorm(in_chans)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        B, H, W, C = x.size()
        shortcut = x
        x = self.norm1(x)
        if self.mode == "block":
            x = window_partition(x, window_size=self.partition_size)
        else:
            x = grid_partition(x, grid_size=self.partition_size)
        x = x.view(-1, self.partition_size * self.partition_size, C)
        attn = self.attn(x)
        if self.mode == "block":
            attn = window_reverse(attn, self.partition_size, H, W)
        else:
            attn = grid_reverse(attn, self.partition_size, H, W)
        x = shortcut + self.drop_path(attn)
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class MaxVitBlock(nn.Module):
    def __init__(
        self,
        in_chans,
        expansion_rate,
        out_chans,
        downsample,
        head_dim=32,
        partition_size=7,
        mlp_ratio=4.0,
        dropout=0.0,
        drop_path=0.0,
    ) -> None:
        super().__init__()

        self.mbconv = MBConv(
            in_chans=in_chans,
            out_chans=out_chans,
            expansion_rate=expansion_rate,
            downsample=downsample,
        )

        self.block_att = TransFormer(
            in_chans=out_chans,
            head_dim=head_dim,
            partition_size=partition_size,
            mlp_ratio=mlp_ratio,
            mode="block",
            drop_rate=dropout,
            drop_path_rate=drop_path,
        )

        self.grid_att = TransFormer(
            in_chans=out_chans,
            head_dim=head_dim,
            partition_size=partition_size,
            mlp_ratio=mlp_ratio,
            mode="grid",
            drop_rate=dropout,
            drop_path_rate=drop_path,
        )

    def forward(self, x):
        x = self.mbconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.block_att(x)
        x = self.grid_att(x)
        x = x.permute(0, 3, 1, 2)
        return x


class MaxVit(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_class=2,
        stem_sizes=[64, 64],
        depths=[2, 2, 5, 2],
        block_sizes=[64, 128, 256, 512],
        drop_out=0.0,
        drop_path=0.0,
    ) -> None:
        super().__init__()

        self.drop_out = drop_out
        self.drop_path = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]

        self.stem = self._make_stem(in_chans, stem_sizes)
        self.block1 = self._make_layers(
            in_chans=stem_sizes[-1],
            out_chans=block_sizes[0],
            depth=depths[0],
            layers_idx=0,
        )
        self.block2 = self._make_layers(
            in_chans=block_sizes[0],
            out_chans=block_sizes[1],
            depth=depths[1],
            layers_idx=sum(depths[:1]),
        )
        self.block3 = self._make_layers(
            in_chans=block_sizes[1],
            out_chans=block_sizes[2],
            depth=depths[2],
            layers_idx=sum(depths[:2]),
        )
        self.block4 = self._make_layers(
            in_chans=block_sizes[2],
            out_chans=block_sizes[3],
            depth=depths[3],
            layers_idx=sum(depths[:3]),
        )

        self.output = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.LayerNorm(block_sizes[-1]),
            nn.Linear(block_sizes[-1], block_sizes[-1], bias=True),
            nn.Tanh(),
            nn.Linear(block_sizes[-1], num_class, bias=True),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.output(x)
        return x

    def _make_stem(self, in_chans, stem_sizes):
        layers = nn.ModuleList([])
        for i in range(len(stem_sizes)):
            if i == 0:
                layers.append(
                    nn.Conv2d(
                        in_chans,
                        stem_sizes[i],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=True,
                    )
                )
            else:
                layers.append(
                    nn.Conv2d(
                        stem_sizes[i - 1],
                        stem_sizes[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=True,
                    )
                )
            if i != len(stem_sizes) - 1:
                layers.append(nn.BatchNorm2d(stem_sizes[i]))
                layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def _make_layers(
        self,
        in_chans,
        out_chans,
        depth,
        expansion_rate=4,
        head_dim=32,
        partition_size=7,
        mlp_ratio=4.0,
        layers_idx=0,
    ):
        layers = nn.ModuleList([])
        for i in range(depth):
            if i == 0:
                layers.append(
                    MaxVitBlock(
                        in_chans=in_chans,
                        expansion_rate=expansion_rate,
                        out_chans=out_chans,
                        downsample=True,
                        head_dim=head_dim,
                        partition_size=partition_size,
                        mlp_ratio=mlp_ratio,
                        dropout=self.drop_out,
                        drop_path=self.drop_path[layers_idx + i],
                    )
                )
            else:
                layers.append(
                    MaxVitBlock(
                        in_chans=out_chans,
                        expansion_rate=expansion_rate,
                        out_chans=out_chans,
                        downsample=False,
                        head_dim=head_dim,
                        partition_size=partition_size,
                        mlp_ratio=mlp_ratio,
                        dropout=self.drop_out,
                        drop_path=self.drop_path[layers_idx + i],
                    )
                )
        return nn.Sequential(*layers)


def maxvit_t(in_chans, num_class):
    model = MaxVit(
        in_chans=in_chans,
        num_class=num_class,
        stem_sizes=[64, 64],
        depths=[2, 2, 5, 2],
        block_sizes=[64, 128, 256, 512],
        drop_out=0.0,
        drop_path=0.0,
    )
    return model

def maxvit_s(in_chans, num_class):
    model = MaxVit(
        in_chans=in_chans,
        num_class=num_class,
        stem_sizes=[64, 64],
        depths=[2, 2, 5, 2],
        block_sizes=[96, 192, 384, 768],
        drop_out=0.0,
        drop_path=0.0,
    )
    return model

def maxvit_b(in_chans, num_class):
    model = MaxVit(
        in_chans=in_chans,
        num_class=num_class,
        stem_sizes=[64, 64],
        depths=[2, 6, 14, 2],
        block_sizes=[96, 192, 384, 768],
        drop_out=0.0,
        drop_path=0.0,
    )
    return model

def maxvit_l(in_chans, num_class):
    model = MaxVit(
        in_chans=in_chans,
        num_class=num_class,
        stem_sizes=[64, 64],
        depths=[2, 6, 14, 2],
        block_sizes=[128, 256, 512, 1024],
        drop_out=0.0,
        drop_path=0.0,
    )
    return model


if __name__ == "__main__":
    # from torchviz import make_dot

    inputs = torch.randn(1, 3, 224, 224)
    # # model = MBConv(in_chans=3, downsample=False, out_chans=64)
    # # model = TransFormer(mode='grid')
    # model = MaxVitBlock(in_chans=32, expansion_rate=4, out_chans=64, downsample=False)
    # outputs = model(inputs)
    # # graph = make_dot(outputs, )
    # # graph.view()
    # print(outputs.shape)
    # inputs = torch.arange(6*6*6).reshape(2, 6, 6, 3)
    # window = window_partition(inputs, 3)
    # grid = grid_partition(inputs, 3)
    # window_r = window_reverse(window, 3, 6, 6)
    # grid_r = grid_reverse(grid, 3, 6, 6)
    # print(torch.equal(inputs, window_r))
    # print(torch.equal(inputs, grid_r))
    # print(inputs[0,:,:,0])
    # print(window[0, : , :, 0])
    # print(grid[0, :, :, 0])
    model = maxvit_t(3, 2)
    outputs = model(inputs)
    print(outputs.shape)

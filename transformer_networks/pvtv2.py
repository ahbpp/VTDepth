import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import typing as tp
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
# from mmdet.models.builder import BACKBONES
# from mmdet.utils import get_root_logger
# from mmcv.runner import load_checkpoint
import math


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()
    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, linear=linear)

    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class Upsampler(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=1, padding=0, scale_factor=2,
                norm_layer=nn.LayerNorm, act_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans * (2 * scale_factor), kernel_size=kernel_size, padding=padding)
        self.scale_factor = scale_factor
        self.act_norm = act_norm
        self.pixelshufle = nn.PixelShuffle(scale_factor)
        if self.act_norm:
            self.act = nn.ReLU()
            self.norm = norm_layer(out_chans)#nn.BatchNorm2d(out_chans)

    def forward(self, x):
        # x = F.interpolate(x,
        #                   scale_factor=self.scale_factor,
        #                   mode="bilinear",
        #                   align_corners=True)
        x = self.proj(x)
        x = self.pixelshufle(x)
        if self.act_norm:
            x = self.act(x)
            x = self.norm(x.transpose(1, -1)).transpose(1, -1)
        return x


class UpsamplerV3(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=1, padding=0, scale_factor=2,
                norm_layer=nn.LayerNorm, act_norm=True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, out_chans, kernel_size=kernel_size, padding=padding)
        self.scale_factor = scale_factor
        self.act_norm = act_norm
        if self.act_norm:
            self.act = nn.ReLU()
            self.norm = norm_layer(out_chans)#nn.BatchNorm2d(out_chans)

    def forward(self, x):
        x = F.interpolate(x,
                          scale_factor=self.scale_factor,
                          mode="bilinear",
                          align_corners=True)
        x = self.proj(x)
        if self.act_norm:
            x = self.act(x)
            x = self.norm(x.transpose(1, -1)).transpose(1, -1)
        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        # self.apply(self._init_weights)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # self.apply(self._init_weights)
        # self.init_weights(pretrained)

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    # def init_weights(self, pretrained=None):
    #     pass
        # if isinstance(pretrained, str):
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x

    def forward_activations(self, x):
        return self.forward_features(x)


class PVTDecoderV2V2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], linear=False, scales=(0, 1, 2, 3)):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(embed_dim) - 1
        self.linear = linear
        self.scales = scales
        self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages, -1, -1):
            in_dim = embed_dim[-1] if i == self.num_stages else self.embed_dim[i]
            block = nn.ModuleList([Block(
                dim=in_dim,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            out_dim = self.embed_dim[i] if i == 0 else self.embed_dim[i-1]
            upsampler = Upsampler(
                            in_chans=in_dim,
                            out_chans=out_dim,
                            kernel_size=3,
                            padding=1,
                            scale_factor=2,
                            act_norm=True,
                            norm_layer=norm_layer)
            block2 = nn.ModuleList([Block(
                dim=out_dim, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            cur += depths[i]
            norm = norm_layer(out_dim)
            dispconv = nn.Sequential(
                nn.Conv2d(out_dim, 1, kernel_size=3, padding=1),
            )

            setattr(self, f"upsample{i}", upsampler)
            if i + 1 in self.scales:
                setattr(self, f"dispconv{i}", dispconv)
            setattr(self, f"block{i}", block)
            setattr(self, f"block_2{i}", block2)
            setattr(self, f"norm{i}", norm)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim[0], 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )


    def forward_features(self, input_features: tp.List[torch.Tensor]):
        B = input_features[0].shape[0]
        outputs = {}
        x = input_features[-1]
        for i in range(self.num_stages, -1, -1):
            block = getattr(self, f"block{i}")
            upsample = getattr(self, f"upsample{i}")
            block2 = getattr(self, f"block_2{i}")
            norm = getattr(self, f"norm{i}")
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = upsample(x)
            if i > 0:
                x += input_features[i - 1]

            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block2:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if i+1 in self.scales:
                dispconv = getattr(self, f"dispconv{i}")
                logits = dispconv(x)
                outputs[("disp", i+1)] = torch.sigmoid(logits)

        logits = self.final_conv(x)
        outputs[("disp", 0)] = torch.sigmoid(logits)
        return outputs


    def forward(self, x):
        return self.forward_features(x)


class PVTDecoderV2V2Res(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], linear=False, scales=(0, 1, 2, 3)):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(embed_dim) - 1
        self.linear = linear
        self.scales = scales
        self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages, -1, -1):
            in_dim = embed_dim[-1] if i == self.num_stages else self.embed_dim[i]
            block = nn.ModuleList([Block(
                dim=in_dim,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            out_dim = self.embed_dim[i] if i == 0 else self.embed_dim[i-1]
            upsampler = Upsampler(
                            in_chans=in_dim,
                            out_chans=out_dim,
                            kernel_size=3,
                            padding=1,
                            scale_factor=2,
                            act_norm=True,
                            norm_layer=norm_layer)
            block2 = nn.ModuleList([Block(
                dim=out_dim, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            cur += depths[i]
            norm = norm_layer(out_dim)
            dispconv = nn.Sequential(
                nn.Conv2d(out_dim, 1, kernel_size=3, padding=1),
            )

            setattr(self, f"upsample{i}", upsampler)
            if i + 1 in self.scales:
                setattr(self, f"dispconv{i}", dispconv)
            setattr(self, f"block{i}", block)
            setattr(self, f"block_2{i}", block2)
            setattr(self, f"norm{i}", norm)
        self.final_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim[0], 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )


    def forward_features(self, input_features: tp.List[torch.Tensor]):
        B = input_features[0].shape[0]
        outputs = {}
        last_features = input_features[0]
        input_features = input_features[1:]
        x = input_features[-1]
        for i in range(self.num_stages, -1, -1):
            block = getattr(self, f"block{i}")
            upsample = getattr(self, f"upsample{i}")
            block2 = getattr(self, f"block_2{i}")
            norm = getattr(self, f"norm{i}")
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = upsample(x)
            if i > 0:
                x += input_features[i - 1]

            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block2:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if i+1 in self.scales:
                dispconv = getattr(self, f"dispconv{i}")
                logits = dispconv(x)
                outputs[("disp", i+1)] = torch.sigmoid(logits)
        logits = self.final_conv(x + last_features)
        outputs[("disp", 0)] = torch.sigmoid(logits)
        return outputs


    def forward(self, x):
        return self.forward_features(x)

class PVTDecoderV2V3(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], linear=False, scales=(0, 1, 2, 3)):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(embed_dim) - 1
        self.linear = linear
        self.scales = scales
        self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(self.num_stages, -1, -1):
            in_dim = embed_dim[-1] if i == self.num_stages else self.embed_dim[i]
            block = nn.ModuleList([Block(
                dim=in_dim,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            out_dim = self.embed_dim[i] if i == 0 else self.embed_dim[i-1]
            upsampler = UpsamplerV3(
                            in_chans=in_dim,
                            out_chans=out_dim,
                            kernel_size=3,
                            padding=1,
                            scale_factor=2,
                            act_norm=True,
                            norm_layer=norm_layer)
            block2 = nn.ModuleList([Block(
                dim=out_dim, num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            cur += depths[i]
            norm = norm_layer(out_dim)
            dispconv = nn.Sequential(
                nn.Conv2d(out_dim, 1, kernel_size=3, padding=1),
            )

            setattr(self, f"upsample{i}", upsampler)
            if i + 1 in self.scales:
                setattr(self, f"dispconv{i}", dispconv)
            setattr(self, f"block{i}", block)
            setattr(self, f"block_2{i}", block2)
            setattr(self, f"norm{i}", norm)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim[0], 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )


    def forward_features(self, input_features: tp.List[torch.Tensor]):
        B = input_features[0].shape[0]
        outputs = {}
        x = input_features[-1]
        for i in range(self.num_stages, -1, -1):
            block = getattr(self, f"block{i}")
            upsample = getattr(self, f"upsample{i}")
            block2 = getattr(self, f"block_2{i}")
            norm = getattr(self, f"norm{i}")
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = upsample(x)
            if i > 0:
                x += input_features[i - 1]

            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block2:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if i+1 in self.scales:
                dispconv = getattr(self, f"dispconv{i}")
                logits = dispconv(x)
                outputs[("disp", i+1)] = torch.sigmoid(logits)

        logits = self.final_conv(x)
        outputs[("disp", 0)] = torch.sigmoid(logits)
        return outputs


    def forward(self, x):
        return self.forward_features(x)

class PVTDecoderV2(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=[32, 64, 128, 256],
                 enc_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, pretrained=None):
        super().__init__()
        # self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.embed_dim = embed_dim

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        for i in range(num_stages):
            in_dim = enc_dims[i + 1] if i == num_stages - 1 else self.embed_dim[i + 1]
            block = nn.ModuleList([Block(
                dim=in_dim,
                num_heads=num_heads[i],
                mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            upsampler = Upsampler(
                            in_chans=in_dim,
                            out_chans=self.embed_dim[i],
                            kernel_size=1,
                            padding=0,
                            scale_factor=2,
                            act_norm=True)
            conv = nn.Sequential(
                nn.GELU(),
                nn.Conv2d(self.embed_dim[i] + enc_dims[i], self.embed_dim[i], 1),
            )
            block2 = nn.ModuleList([Block(
                dim=self.embed_dim[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            cur += depths[i]
            norm = norm_layer(self.embed_dim[i])
            dispconv = nn.Sequential(
                nn.Conv2d(self.embed_dim[i], 1, kernel_size=3, padding=1),
            )

            setattr(self, f"upsample{i + 1}", upsampler)
            if i != num_stages - 1:
                setattr(self, f"dispconv{i + 1}", dispconv)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"conv{i + 1}", conv)
            setattr(self, f"block_2{i + 1}", block2)
            setattr(self, f"norm{i + 1}", norm)

        self.final_conv = nn.Sequential(
            nn.Conv2d(self.embed_dim[0], 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
        )


    def forward_features(self, input_features: tp.List[torch.Tensor]):
        B = input_features[0].shape[0]
        outs = []
        x = input_features[self.num_stages]
        for i in range(self.num_stages):
            block = getattr(self, f"block{self.num_stages - i}")
            conv = getattr(self, f"conv{self.num_stages - i}")
            block2 = getattr(self, f"block_2{self.num_stages - i}")
            norm = getattr(self, f"norm{self.num_stages - i}")
            upsample = getattr(self, f"upsample{self.num_stages - i}")
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            x = upsample(x)
            x = conv(torch.cat((x, input_features[self.num_stages - i - 1]), dim=1))


            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2)
            for blk in block2:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward(self, x):
        self.outputs = {}
        # x = [F.interpolate(el, scale_factor=2, mode="bilinear", align_corners=True) for el in x]# + [x[-1]]
        features = self.forward_features(x)
        for i in range(1, self.num_stages):
            dispconv = getattr(self, f"dispconv{self.num_stages - i}")
            logits = dispconv(features[i])
            self.outputs[("disp", self.num_stages - i)] = torch.sigmoid(logits)
        logits = self.final_conv(features[-1])
        self.outputs[("disp", 0)] = torch.sigmoid(logits)
        return self.outputs


class PVTMiddle(nn.Module):
    def __init__(self, img_size=224, patch_size=3, patch_stride=2, in_chans=512, embed_dim=512,
                 num_heads=8, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depth=2,
                 sr_ratio=1, linear=False, pretrained=None):
        super().__init__()
        self.depth = depth
        self.linear = linear
        dpr = drop_path_rate

        self.patch_embed = OverlapPatchEmbed(img_size=img_size,
                                        patch_size=patch_size,
                                        stride=patch_stride,
                                        in_chans=in_chans,
                                        embed_dim=embed_dim)

        self.block = nn.ModuleList([Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr, norm_layer=norm_layer,
                sr_ratio=sr_ratio, linear=linear)
                for j in range(depth)])
        self.norm = norm_layer(embed_dim)

    def forward(self, x):
        B = x.shape[0]
        x, H, W = self.patch_embed(x)
        for blk in self.block:
            x = blk(x, H, W)
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class PVTEncoderV2(nn.Module):
    def __init__(self, img_size=224, patch_size=15, patch_stride=7,
                 in_chans=3, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, linear=False, num_classes=None, pretrained=None):
        super().__init__()
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        self.linear = linear
        self.embed_dims = embed_dims

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            if i == 0:
                in_ch = in_chans
            elif i == 1:
                in_ch = embed_dims[i - 1] * 2
            else:
                in_ch = embed_dims[i - 1]
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=patch_size if i == 0 else 3,
                                            stride=patch_stride if i == 0 else 2,
                                            in_chans=in_ch,
                                            embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j], norm_layer=norm_layer,
                sr_ratio=sr_ratios[i], linear=linear)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)
        if self.num_classes is not None:
            # classification head
            self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.head = None

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B, in_ch, H, W = x.shape
        outs = []
        x = x.view(-1, self.in_chans, H, W)

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs


    def forward(self, x):
        B, in_chans, H, W = x.shape

        x = self.forward_features(x)
        if self.head is not None:
            x = x[-1].mean((-2, -1))
            x = self.head(x)
            return x

        return x

    def forward_activations(self, x):
        return self.forward_features(x)


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

#
# @BACKBONES.register_module()
# class pvt_v2_b0(PyramidVisionTransformerV2):
#     def __init__(self, **kwargs):
#         super(pvt_v2_b0, self).__init__(
#             patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])
#
#
# @BACKBONES.register_module()
class pvt_v2_b1(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b1, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
#
#
# @BACKBONES.register_module()
class pvt_v2_b2(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)
#
#
# @BACKBONES.register_module()
class pvt_v2_b2_li(PyramidVisionTransformerV2):
    def __init__(self, **kwargs):
        super(pvt_v2_b2_li, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1, linear=True)
#
#
# @BACKBONES.register_module()
# class pvt_v2_b3(PyramidVisionTransformerV2):
#     def __init__(self, **kwargs):
#         super(pvt_v2_b3, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])
#
#
# @BACKBONES.register_module()
# class pvt_v2_b4(PyramidVisionTransformerV2):
#     def __init__(self, **kwargs):
#         super(pvt_v2_b4, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])
#
#
# @BACKBONES.register_module()
# class pvt_v2_b5(PyramidVisionTransformerV2):
#     def __init__(self, **kwargs):
#         super(pvt_v2_b5, self).__init__(
#             patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
#             qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
#             drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'])
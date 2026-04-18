#!/usr/bin/env python
# -*- coding:utf-8 -*-

import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=1, activation=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation:
            x = self.relu(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3, stride=1, padding=1, activation=False))
        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(cout)

    def forward(self, x):
        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)
        return x


# ============================================================
# IMPROVEMENT 3: ResNet50 Pretrained Encoder
#
# Replaces the original shallow U_encoder (3 DoubleConv blocks,
# trained from scratch) with a pretrained ResNet50 backbone.
#
# Why this helps:
#   - ResNet50 pretrained on ImageNet gives rich, generalizable
#     feature representations out of the box.
#   - Medical datasets are small — pretrained features prevent
#     overfitting and speed up convergence significantly.
#   - Deeper residual connections capture more complex textures
#     than the original 3-block encoder.
#
# What gets extracted:
#   ResNet50 Stage    Output size (224x224 input)   Channels
#   ──────────────────────────────────────────────────────────
#   stage0 (stem)     112 × 112                     64
#   layer1            56  × 56                      256
#   layer2            28  × 28                      512  → MTM
#
# Adapter 1x1 convolutions map ResNet channels to what
# U_decoder expects: [64, 128, 256] at [224, 112, 56].
# The FC head of ResNet50 is removed entirely.
# ============================================================
class ResNet50Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50Encoder, self).__init__()

        resnet = models.resnet50(pretrained=pretrained)

        # Stage 0: conv1+bn1+relu+maxpool → (B, 64, 112, 112)
        self.stage0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        # Stage 1: layer1 → (B, 256, 56, 56)
        self.stage1 = resnet.layer1

        # Stage 2: layer2 → (B, 512, 28, 28) — feeds into MTM
        self.stage2 = resnet.layer2

        # Adapter convolutions: ResNet channels → U_decoder expected channels
        # skip0: (B, 64,  112,112) → adapt+upsample → (B, 64,  224,224)
        self.adapt0 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU())

        # skip1: (B, 256, 56, 56) → adapt+upsample → (B, 128, 112,112)
        self.adapt1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU())

        # skip2: (B, 512, 28, 28) → adapt+upsample → (B, 256, 56, 56)
        self.adapt2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU())

        # Final projection: (B, 512, 28,28) → (B, 256, 28,28) for MTM input
        self.proj = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU())

    def forward(self, x):
        """
        Returns:
            out:      (B, 256, 28, 28) — fed into MTM transformer blocks
            features: 3 skip tensors matching original U_encoder outputs
                      [0]: (B,  64, 224, 224)
                      [1]: (B, 128, 112, 112)
                      [2]: (B, 256,  56,  56)
        """
        features = []

        s0 = self.stage0(x)                                          # (B, 64,  112, 112)
        s0_skip = F.interpolate(self.adapt0(s0), scale_factor=2,
                                mode='bilinear', align_corners=False) # (B, 64,  224, 224)
        features.append(s0_skip)

        s1 = self.stage1(s0)                                          # (B, 256, 56,  56)
        s1_skip = F.interpolate(self.adapt1(s1), scale_factor=2,
                                mode='bilinear', align_corners=False) # (B, 128, 112, 112)
        features.append(s1_skip)

        s2 = self.stage2(s1)                                          # (B, 512, 28,  28)
        s2_skip = F.interpolate(self.adapt2(s2), scale_factor=2,
                                mode='bilinear', align_corners=False) # (B, 256, 56,  56)
        features.append(s2_skip)

        out = self.proj(s2)                                           # (B, 256, 28,  28)
        return out, features


# ============================================================
# U_decoder — unchanged, still expects features [64, 128, 256]
# Now receives ResNet50Encoder's adapted skip connections
# ============================================================
class U_decoder(nn.Module):
    def __init__(self):
        super(U_decoder, self).__init__()
        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.res1 = DoubleConv(512, 256)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.res2 = DoubleConv(256, 128)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.res3 = DoubleConv(128, 64)

    def forward(self, x, feature):
        x = self.trans1(x)
        f2 = F.interpolate(feature[2], size=x.shape[2:], mode='bilinear', align_corners=False) if feature[2].shape[2:] != x.shape[2:] else feature[2]
        x = torch.cat((f2, x), dim=1)
        x = self.res1(x)
        x = self.trans2(x)
        f1 = F.interpolate(feature[1], size=x.shape[2:], mode='bilinear', align_corners=False) if feature[1].shape[2:] != x.shape[2:] else feature[1]
        x = torch.cat((f1, x), dim=1)
        x = self.res2(x)
        x = self.trans3(x)
        f0 = F.interpolate(feature[0], size=x.shape[2:], mode='bilinear', align_corners=False) if feature[0].shape[2:] != x.shape[2:] else feature[0]
        x = torch.cat((f0, x), dim=1)
        x = self.res3(x)
        return x


class MEAttention(nn.Module):
    def __init__(self, dim, configs):
        super(MEAttention, self).__init__()
        self.num_heads = configs["head"]
        self.coef = 4
        self.query_liner = nn.Linear(dim, dim * self.coef)
        self.num_heads = self.coef * self.num_heads
        self.k = 256 // self.coef
        self.linear_0 = nn.Linear(dim * self.coef // self.num_heads, self.k)
        self.linear_1 = nn.Linear(self.k, dim * self.coef // self.num_heads)
        self.proj = nn.Linear(dim * self.coef, dim)

    def forward(self, x):
        B, N, C = x.shape
        x = self.query_liner(x)
        x = x.view(B, N, self.num_heads, -1).permute(0, 2, 1, 3)
        attn = self.linear_0(x)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        x = self.linear_1(attn).permute(0, 2, 1, 3).reshape(B, N, -1)
        x = self.proj(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, configs, axial=False):
        super(Attention, self).__init__()
        self.axial = axial
        self.dim = dim
        self.num_head = configs["head"]
        self.attention_head_size = int(self.dim / configs["head"])
        self.all_head_size = self.num_head * self.attention_head_size
        self.query_layer = nn.Linear(self.dim, self.all_head_size)
        self.key_layer = nn.Linear(self.dim, self.all_head_size)
        self.value_layer = nn.Linear(self.dim, self.all_head_size)
        self.out = nn.Linear(self.dim, self.dim)
        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_head, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x

    def forward(self, x):
        if self.axial:
            b, h, w, c = x.shape
            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)
            query_layer_x = mixed_query_layer.view(b * h, w, -1)
            key_layer_x = mixed_key_layer.view(b * h, w, -1).transpose(-1, -2)
            attention_scores_x = torch.matmul(query_layer_x, key_layer_x)
            attention_scores_x = attention_scores_x.view(b, -1, w, w)
            query_layer_y = mixed_query_layer.permute(0, 2, 1, 3).contiguous().view(b * w, h, -1)
            key_layer_y = mixed_key_layer.permute(0, 2, 1, 3).contiguous().view(b * w, h, -1).transpose(-1, -2)
            attention_scores_y = torch.matmul(query_layer_y, key_layer_y)
            attention_scores_y = attention_scores_y.view(b, -1, h, h)
            return attention_scores_x, attention_scores_y, mixed_value_layer
        else:
            mixed_query_layer = self.query_layer(x)
            mixed_key_layer = self.key_layer(x)
            mixed_value_layer = self.value_layer(x)
            query_layer = self.transpose_for_scores(mixed_query_layer).permute(0, 1, 2, 4, 3, 5).contiguous()
            key_layer = self.transpose_for_scores(mixed_key_layer).permute(0, 1, 2, 4, 3, 5).contiguous()
            value_layer = self.transpose_for_scores(mixed_value_layer).permute(0, 1, 2, 4, 3, 5).contiguous()
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            atten_probs = self.softmax(attention_scores)
            context_layer = torch.matmul(atten_probs, value_layer)
            context_layer = context_layer.permute(0, 1, 2, 4, 3, 5).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            attention_output = self.out(context_layer)
        return attention_output


# ============================================================
# IMPROVEMENT 1: Adaptive Window Size
# ============================================================
class AdaptiveWindowSelector(nn.Module):
    def __init__(self, dim, window_sizes=[2, 4, 8]):
        super(AdaptiveWindowSelector, self).__init__()
        self.window_sizes = window_sizes
        self.edge_conv = nn.Conv2d(dim, 1, kernel_size=3, padding=1, bias=False)
        self.weight_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, len(window_sizes)), nn.Softmax(dim=-1))

    def forward(self, x_2d):
        edge_map = torch.abs(self.edge_conv(x_2d))
        avg_edge = F.avg_pool2d(edge_map, kernel_size=3, stride=1, padding=1)
        avg_edge = avg_edge.permute(0, 2, 3, 1)
        weights = self.weight_mlp(avg_edge)
        return weights


class WinAttention(nn.Module):
    def __init__(self, configs, dim, window_size=None):
        super(WinAttention, self).__init__()
        self.window_size = window_size if window_size is not None else configs["win_size"]
        self.attention = Attention(dim, configs)

    def forward(self, x):
        b, n, c = x.shape
        h, w = int(np.sqrt(n)), int(np.sqrt(n))
        x = x.permute(0, 2, 1).contiguous().view(b, c, h, w)
        if h % self.window_size != 0:
            right_size = h + self.window_size - h % self.window_size
            new_x = torch.zeros((b, c, right_size, right_size)).to(x.device)
            new_x[:, :, 0:x.shape[2], 0:x.shape[3]] = x[:]
            new_x[:, :, x.shape[2]:, x.shape[3]:] = x[:, :,
                                                       (x.shape[2] - right_size):,
                                                       (x.shape[3] - right_size):]
            x = new_x
            b, c, h, w = x.shape
        x = x.view(b, c, h // self.window_size, self.window_size,
                   w // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(
            b, h // self.window_size, w // self.window_size,
            self.window_size * self.window_size, c).cuda()
        x = self.attention(x)
        return x


class DlightConv(nn.Module):
    def __init__(self, dim, configs):
        super(DlightConv, self).__init__()
        self.linear = nn.Linear(dim, configs["win_size"] * configs["win_size"])
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        h = x
        avg_x = torch.mean(x, dim=-2)
        x_prob = self.softmax(self.linear(avg_x))
        x = torch.mul(h, x_prob.unsqueeze(-1))
        x = torch.sum(x, dim=-2)
        return x


class GaussianTrans(nn.Module):
    def __init__(self):
        super(GaussianTrans, self).__init__()
        self.bias = nn.Parameter(-torch.abs(torch.randn(1)))
        self.shift = nn.Parameter(torch.abs(torch.randn(1)))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x, atten_x_full, atten_y_full, value_full = x
        new_value_full = torch.zeros_like(value_full)
        for r in range(x.shape[1]):
            for c in range(x.shape[2]):
                atten_x = atten_x_full[:, r, c, :]
                atten_y = atten_y_full[:, c, r, :]
                dis_x = torch.tensor([(h - c)**2 for h in range(x.shape[2])]).cuda()
                dis_y = torch.tensor([(w - r)**2 for w in range(x.shape[1])]).cuda()
                dis_x = -(self.shift * dis_x + self.bias).cuda()
                dis_y = -(self.shift * dis_y + self.bias).cuda()
                atten_x = self.softmax(dis_x + atten_x)
                atten_y = self.softmax(dis_y + atten_y)
                new_value_full[:, r, c, :] = torch.sum(
                    atten_x.unsqueeze(dim=-1) * value_full[:, r, :, :] +
                    atten_y.unsqueeze(dim=-1) * value_full[:, :, c, :],
                    dim=-2)
        return new_value_full


class CSAttention(nn.Module):
    def __init__(self, dim, configs):
        super(CSAttention, self).__init__()
        self.configs = configs
        self.window_sizes = [2, 4, 8]
        self.win_attens = nn.ModuleList([
            WinAttention(configs, dim, window_size=ws) for ws in self.window_sizes])
        self.adaptive_selector = AdaptiveWindowSelector(dim, self.window_sizes)
        self.dlightconv = DlightConv(dim, configs)
        self.global_atten = Attention(dim, configs, axial=True)
        self.gaussiantrans = GaussianTrans()
        self.up = nn.UpsamplingBilinear2d(scale_factor=4)
        self.queeze = nn.Conv2d(2 * dim, dim, 1)

    def forward(self, x):
        origin_size = x.shape
        B = origin_size[0]
        origin_h = int(np.sqrt(origin_size[1]))
        origin_w = origin_h
        C = origin_size[2]

        scale_outputs = []
        for win_atten in self.win_attens:
            out = win_atten(x)
            b, p1, p2, win, c = out.shape
            out_spatial = out.view(b, p1, p2, int(np.sqrt(win)), int(np.sqrt(win)), c)
            out_spatial = out_spatial.permute(0, 1, 3, 2, 4, 5).contiguous()
            out_spatial = out_spatial.view(b, p1 * int(np.sqrt(win)), p2 * int(np.sqrt(win)), c)
            out_spatial = out_spatial[:, :origin_h, :origin_w, :]
            scale_outputs.append(out_spatial)

        x_2d = x.view(B, origin_h, origin_w, C).permute(0, 3, 1, 2)
        blend_weights = self.adaptive_selector(x_2d)
        blended = torch.zeros(B, origin_h, origin_w, C).to(x.device)
        for i, scale_out in enumerate(scale_outputs):
            w = blend_weights[:, :, :, i:i+1]
            blended = blended + w * scale_out
        h = blended.permute(0, 3, 1, 2).contiguous()

        x_win = self.win_attens[1](x)
        x_dl = self.dlightconv(x_win)
        atten_x, atten_y, mixed_value = self.global_atten(x_dl)
        gaussian_input = (x_dl, atten_x, atten_y, mixed_value)
        x_g = self.gaussiantrans(gaussian_input)
        x_g = x_g.permute(0, 3, 1, 2).contiguous()
        x_g = self.up(x_g)

        if x_g.shape[2] != h.shape[2] or x_g.shape[3] != h.shape[3]:
            x_g = F.interpolate(x_g, size=(h.shape[2], h.shape[3]),
                                mode='bilinear', align_corners=False)
        x_out = self.queeze(torch.cat((x_g, h), dim=1))
        x_out = x_out.permute(0, 2, 3, 1).contiguous()
        x_out = x_out[:, :origin_h, :origin_w, :].contiguous()
        x_out = x_out.view(B, -1, C)
        return x_out


class EAmodule(nn.Module):
    def __init__(self, dim):
        super(EAmodule, self).__init__()
        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.CSAttention = CSAttention(dim, configs)
        self.EAttention = MEAttention(dim, configs)

    def forward(self, x):
        h = x
        x = self.SlayerNorm(x)
        x = self.CSAttention(x)
        x = h + x
        h = x
        x = self.ElayerNorm(x)
        x = self.EAttention(x)
        x = h + x
        return x


class DecoderStem(nn.Module):
    def __init__(self):
        super(DecoderStem, self).__init__()
        self.block = U_decoder()

    def forward(self, x, features):
        x = self.block(x, features)
        return x


# ============================================================
# Stem: now wraps ResNet50Encoder instead of U_encoder.
# Interface is identical — returns (x, features) same as before.
# ============================================================
class Stem(nn.Module):
    def __init__(self):
        super(Stem, self).__init__()
        self.model = ResNet50Encoder(pretrained=True)       # CHANGED
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0)
        self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))

    def forward(self, x):
        x, features = self.model(x)   # (B, 256, 28, 28), [skip0, skip1, skip2]
        x = self.trans_dim(x)         # (B, 256, 28, 28)
        x = x.flatten(2)              # (B, 256, 784)
        x = x.transpose(-2, -1)       # (B, 784, 256)
        x = x + self.position_embedding
        return x, features


class encoder_block(nn.Module):
    def __init__(self, dim):
        super(encoder_block, self).__init__()
        self.block = nn.ModuleList([
            EAmodule(dim),
            EAmodule(dim),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
        ])

    def forward(self, x):
        x = self.block[0](x)
        x = self.block[1](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        skip = x
        x = self.block[2](x)
        return x, skip


class decoder_block(nn.Module):
    def __init__(self, dim, flag):
        super(decoder_block, self).__init__()
        self.flag = flag
        if not self.flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2, padding=0),
                nn.Conv2d(dim, dim // 2, kernel_size=1, stride=1),
                EAmodule(dim // 2),
                EAmodule(dim // 2)
            ])
        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2, padding=0),
                EAmodule(dim),
                EAmodule(dim)
            ])

    def forward(self, x, skip):
        if not self.flag:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[2](x)
            x = self.block[3](x)
        else:
            x = self.block[0](x)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            x = self.block[2](x)
        return x


class MTUNet(nn.Module):
    def __init__(self, out_ch=4):
        super(MTUNet, self).__init__()
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.bottleneck = nn.Sequential(
            EAmodule(configs["bottleneck"]),
            EAmodule(configs["bottleneck"]))
        self.decoder = nn.ModuleList()
        self.decoder_stem = DecoderStem()

        for i in range(len(configs["encoder"])):
            dim = configs["encoder"][i]
            self.encoder.append(encoder_block(dim))
        for i in range(len(configs["decoder"]) - 1):
            dim = configs["decoder"][i]
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(decoder_block(configs["decoder"][-1], True))
        self.SegmentationHead = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)
        skips = []
        for i in range(len(self.encoder)):
            x, skip = self.encoder[i](x)
            skips.append(skip)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        x = self.bottleneck(x)
        B, N, C = x.shape
        x = x.view(B, int(np.sqrt(N)), -1, C).permute(0, 3, 1, 2)
        for i in range(len(self.decoder)):
            x = self.decoder[i](x, skips[len(self.decoder) - i - 1])
            B, N, C = x.shape
            x = x.view(B, int(np.sqrt(N)), int(np.sqrt(N)), C).permute(0, 3, 1, 2)
        x = self.decoder_stem(x, features)
        x = self.SegmentationHead(x)
        return x


configs = {
    "win_size": 4,
    "head": 8,
    "axis": [28, 16, 8],
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
    "decoder_stem": [(256, 512), (256, 256), (128, 64), 32]
}
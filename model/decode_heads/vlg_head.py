# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat

from mmseg.models.builder import HEADS
from third_party.maskclip.models.backbones.maskclip_vit import TransformerEncoderLayer
from model.text_embeddings import aggregate_concept_predictions, get_class_to_concept_idxs
    

class SemanticTransformer(nn.Module):
    def __init__(self, channels, text_channels, num_heads, pool_size) -> None:
        super().__init__()
        if pool_size is not None:
            self.pool = nn.AvgPool2d(pool_size)
        else:
            self.pool = None
        self.transformer = TransformerEncoderLayer(
            embed_dims=channels+text_channels,
            num_heads=num_heads,
            feedforward_channels=4*channels)

    def forward(self, x, text_feats):
        B, C, _, H, W = x.shape
        if self.pool is None:
            x_pool = x
        else:
            x_pool = rearrange(x, 'b c n h w -> (b n) c h w')
            x_pool = self.pool(x_pool)
            x_pool = rearrange(x_pool, '(b n) c h w -> b c n h w', b=B)
        _, _, _, H_pool, W_pool = x_pool.shape

        x_pool = rearrange(x_pool, 'b c n h w -> (b h w) n c')
        if text_feats is not None:
            text_feats = repeat(text_feats, 'b n c -> (b h w) n c', h=H_pool, w=W_pool)
            x_pool = torch.cat([x_pool, text_feats], dim=-1)

        x_pool, _, _, _ = self.transformer(x_pool)
        
        if text_feats is not None:
            x_pool = x_pool[..., :C]  # discard text tokens

        if self.pool is None:
            x_pool = rearrange(x_pool, '(b h w) n c -> b c n h w', b=B, h=H_pool, w=W_pool)
        else:
            x_pool = rearrange(x_pool, '(b h w) n c -> (b n) c h w', h=H_pool, w=W_pool)
            x_pool = F.interpolate(x_pool, size=(H, W), mode='bilinear', align_corners=True)
            x_pool = rearrange(x_pool, '(b n) c h w -> b c n h w', b=B)

        x = x + x_pool
        return x


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.GroupNorm(out_channels // 16, out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates=(1, 6, 12, 18), out_channels=None):
        super(ASPPModule, self).__init__()
        if out_channels is None:
            out_channels = in_channels

        self.aspp_convs = nn.ModuleList()
        for dilation in atrous_rates:
            ksize = 1 if dilation == 1 else 3
            padding = 0 if dilation == 1 else dilation
            self.aspp_convs.append(
                nn.Sequential(nn.Conv2d(in_channels, out_channels, ksize, padding=padding,
                                        dilation=dilation, bias=False),
                              nn.GroupNorm(out_channels // 16, out_channels),
                              nn.ReLU(True))
            )
        self.aspp_convs.append(ASPPPooling(in_channels, out_channels))

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.GroupNorm(out_channels // 16, out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feats = []
        for c in self.aspp_convs:
            feats.append(c(x))
        y = torch.cat(feats, 1)
        y = self.project(y)
        y = x + y
        return y


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels - skip_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_x=None):
        x = self.up(x)
        if skip_x is not None:
            N = x.size(0) // skip_x.size(0)
            skip_x = F.interpolate(skip_x, size=x.shape[-2:], mode='bilinear', align_corners=True)
            skip_x = repeat(skip_x, "b c h w -> (b n) c h w", n=N)
            x = torch.cat([x, skip_x], dim=1)
        return self.conv(x)


@HEADS.register_module()
class VLGHead(nn.Module):
    def __init__(self,
        img_size,
        num_classes,
        text_in_channels,
        text_channels,
        up_channels,
        skip_in_channels,
        skip_channels,
        skip_from_conv_feat,
        num_layers,
        num_heads,
        channels,
        pool_size,
        conv1_ksize,
        loss_decode,
        align_corners,
    ) -> None:
        super().__init__()
        self.image_size = img_size
        self.num_classes = num_classes
        self.align_corners = align_corners
        self.text_in_channels = text_in_channels
        self.num_layers = num_layers
        self.channels = channels
        self.skip_from_conv_feat = skip_from_conv_feat
        assert loss_decode is None

        self.conv1 = nn.Conv2d(1, channels, kernel_size=conv1_ksize, stride=1, padding=(conv1_ksize-1)//2)
        self.aspp = ASPPModule(channels)
        self.layers = nn.ModuleList([
            SemanticTransformer(
                channels=channels, text_channels=text_channels, num_heads=num_heads, pool_size=pool_size
            ) for _ in range(num_layers)
        ])

        self.text_proj = nn.Sequential(
            nn.Linear(text_in_channels, text_channels),
            nn.ReLU())

        self.skip_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(sic, sc, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
            ) for sic, sc in zip(skip_in_channels, skip_channels)
        ])

        self.up1 = Up(channels, up_channels[0], skip_channels[0])
        self.up2 = Up(up_channels[0], up_channels[1], skip_channels[1])
        self.head = nn.Conv2d(up_channels[1], 1, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs, force_output_pred_masks=False):
        inputs_both = inputs
        img_feat_pyramid = inputs_both[0][0]
        img_feats = img_feat_pyramid[-1]
        if self.skip_from_conv_feat:
            conv_feats = inputs_both[2]
            if len(img_feat_pyramid) > 1:
                skip_feats = [
                    *img_feat_pyramid[:-1][::-1],
                    *conv_feats[::-1],
                ]
            else:
                skip_feats = conv_feats[::-1]
            assert len(self.skip_proj) == len(skip_feats)
        else:
            skip_feats = img_feat_pyramid[:-1][::-1]
        text_feats = inputs_both[1]

        text_feats = text_feats.repeat(img_feats.shape[0], 1, 1).float()
        B, C, H, W = img_feats.shape
        assert list(text_feats.shape) == [B, self.num_classes, C]

        # Compute Similarity Map
        img_feats = F.normalize(img_feats, dim=1)
        text_feats = F.normalize(text_feats, dim=-1)
        x = torch.einsum('bchw, bnc -> bnhw', img_feats, text_feats)

        # Spatial Reasoning
        x = rearrange(x, 'b n h w -> (b n) () h w')
        x = self.conv1(x)
        x = self.aspp(x)
        x = rearrange(x, '(b n) c h w -> b c n h w', b=B)

        # Semantic Reasoning
        if self.text_proj is not None:
            text_feats = self.text_proj(text_feats)

        for layer in self.layers:
            x = layer(x, text_feats)

        # Upsampling
        if self.skip_proj is not None:
            skip_feats = [proj(f) for proj, f in zip(self.skip_proj, skip_feats)]

        x = rearrange(x, 'b c n h w -> (b n) c h w')
        x = self.up1(x, skip_feats[0])
        x = self.up2(x, skip_feats[1])
        x = self.head(x)
        x = rearrange(x, '(b n) () h w -> b n h w', b=B)

        if x.shape[1] != self.num_classes:
            cls2con = get_class_to_concept_idxs(self.load_text_embedding)
            x = aggregate_concept_predictions(x, cls2con)

        if force_output_pred_masks:
            x = F.interpolate(x, size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=self.align_corners)
            x = {"pred_masks": x}

        return x

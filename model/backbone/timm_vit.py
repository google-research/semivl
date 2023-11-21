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
import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch import nn
from torch.nn import functional as F
from mmseg.models.builder import BACKBONES




@BACKBONES.register_module()
class TIMMVisionTransformer(nn.Module):

    def __init__(
        self,
        variant,
        timm_load_pretrained,
        drop_path_rate,
        img_size,
        out_indices,
    ):
        super(TIMMVisionTransformer, self).__init__()
        self.m = timm.create_model(
            variant,
            pretrained=timm_load_pretrained,
            drop_path_rate=drop_path_rate,
            img_size=img_size,
        )
        self.patch_size = self.m.patch_embed.patch_size
        self.img_size = img_size
        self.out_indices = out_indices
        assert max(self.out_indices) <= 11

    def forward_features(self, x):
        feats = []
        x = self.m.patch_embed(x)
        x = self.m._pos_embed(x)
        if self.m.grad_checkpointing and not torch.jit.is_scripting():
            raise ValueError(self.m.grad_checkpointing)
            # x = checkpoint_seq(self.blocks, x)
        else:
            for i, block in enumerate(self.m.blocks):
                x = block(x)
                if i in self.out_indices:
                    out = self.m.norm(x)
                    feats.append(out)
        x = self.m.norm(x)
        return x, feats

    def forward(self, x: torch.Tensor):
        if x.shape[-2] != self.m.patch_embed.img_size[0] or x.shape[-1] != self.m.patch_embed.img_size[1]:
            assert not self.training
            x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=False)
        B, _, H, W = x.shape
        H = H // self.patch_size[0]
        W = W // self.patch_size[1]

        x, feats = self.forward_features(x)
        outs = [
            tuple([f[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2) for f in feats]),
            x[:, 0],  # cls_token
        ]

        return outs

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
from torch.nn import functional as F

from mmseg.models.builder import HEADS

from third_party.unimatch.model.semseg.deeplabv3plus import ASPPModule
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class DLV3PHead(BaseDecodeHead):

    def __init__(self, c1_in_channels, c1_channels, dilations, img_size, **kwargs):
        super(DLV3PHead, self).__init__(**kwargs)
        self.image_size = img_size
        self.aspp = ASPPModule(self.in_channels, dilations)
        self.c1_proj = nn.Sequential(
            nn.Conv2d(c1_in_channels, c1_channels, 1, bias=False),
            nn.BatchNorm2d(c1_channels),
            nn.ReLU(True))
        fuse_channels = self.in_channels // 8 + c1_channels
        self.head = nn.Sequential(
            nn.Conv2d(fuse_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, self.num_classes, 1, bias=True))
        self.conv_seg = None

    def forward(self, inputs, force_output_pred_masks=False):
        if force_output_pred_masks:
            inputs = inputs[0][0]
        assert len(inputs) == 2
        c1, c4 = inputs[0], inputs[1]

        c4 = self.aspp(c4)
        c1 = self.c1_proj(c1)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=self.align_corners)
        x = torch.cat([c1, c4], dim=1)
        out = self.head(x)

        if force_output_pred_masks:
            out = F.interpolate(out, size=(self.image_size, self.image_size),
                                mode='bilinear', align_corners=self.align_corners)
            out = {"pred_masks": out}

        return out

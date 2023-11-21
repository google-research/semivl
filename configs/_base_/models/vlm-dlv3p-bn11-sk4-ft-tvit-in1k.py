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


norm_cfg = dict(type='SyncBN', requires_grad=True)
img_size = 512

model = dict(
    type='VLM',
    backbone=dict(
        type='TIMMVisionTransformer',
        variant='vit_base_patch16_224',
        timm_load_pretrained=True,
        drop_path_rate=0.1,
        img_size=img_size,
        out_indices=[4, 11]),
    decode_head=dict(
        type='DLV3PHead',
        img_size=img_size,
        in_channels=768,
        in_index=3,
        channels=256,
        dilations=(6, 12, 18),
        c1_in_channels=768,
        c1_channels=48,
        dropout_ratio=0,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        init_cfg=None,
    ),
    freeze_backbone=False,
    exclude_keys=None,
)
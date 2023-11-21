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


img_size = 512

model = dict(
    type='VLM',
    pretrained='pretrained/clip2mmseg_ViT16_clip_backbone.pth',
    backbone=dict(
        type='MaskClipVisionTransformer',
        img_size=(img_size, img_size),
        patch_size=16,
        patch_bias=False,
        in_channels=3,
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        out_indices=[0, 4, 12],
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=False,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        patch_norm=False,
        pre_norm=True,
        final_norm=True,
        return_clip_embed=True,
        return_qkv=True,
        interpolate_mode='bicubic',
        num_fcs=2,
        norm_eval=False
    ),
    decode_head=dict(
        type='VLGHead',
        img_size=img_size,
        num_classes=19,
        text_in_channels=512,
        text_channels=128,
        up_channels=(64, 32),
        skip_in_channels=(768, 768),
        skip_channels=(32, 16),
        skip_from_conv_feat=False,
        num_layers=2,
        num_heads=4,
        channels=128,
        pool_size=(4, 4),
        conv1_ksize=7,
        align_corners=False,
        loss_decode=None,
    ),
    freeze_backbone=True,
    exclude_keys=['attn', 'pos_embed'],
)

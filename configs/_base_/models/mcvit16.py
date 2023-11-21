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
backbone=dict(
    type='MaskClipVisionTransformer',
    pretrained='pretrained/clip2mmseg_ViT16_clip_backbone.pth',
    img_size=(img_size, img_size),
    patch_size=16,
    patch_bias=False,
    in_channels=3,
    embed_dims=768,
    num_layers=12,
    num_heads=12,
    mlp_ratio=4,
    out_indices=None,
    qkv_bias=True,
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.0,
    with_cls_token=True,
    output_cls_token=False,
    norm_cfg=dict(type='LN', eps=1e-6),
    act_cfg=dict(type='GELU'),
    patch_norm=False,
    pre_norm = True,
    final_norm=True,
    return_clip_embed=True,
    return_qkv=True,
    interpolate_mode='bicubic',
    num_fcs=2,
    norm_eval=False
)

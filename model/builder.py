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


import types
from functools import reduce

import torch
from mmcv.utils import Config
from mmseg.models import ASPPHead, DepthwiseSeparableASPPHead, build_segmentor
from mmseg.ops import resize
from torch.nn import functional as F

from model.backbone.timm_vit import TIMMVisionTransformer
from model.decode_heads.dlv3p_head import DLV3PHead
from model.decode_heads.vlg_head import VLGHead
from model.vlm import VLM
from third_party.maskclip.models.backbones.maskclip_vit import MaskClipVisionTransformer
from third_party.maskclip.models.decode_heads.maskclip2_head import MaskClip2Head
from third_party.maskclip.models.decode_heads.maskclip_head import MaskClipHead
from third_party.unimatch.model.semseg.deeplabv3plus import DeepLabV3Plus
from third_party.zegclip.losses.atm_loss import SegLossPlus
from third_party.zegclip.models.backbones.clip_vit import CLIPVisionTransformer
from third_party.zegclip.models.backbones.clip_vpt_vit import VPTCLIPVisionTransformer
from third_party.zegclip.models.backbones.text_encoder import CLIPTextEncoder
from third_party.zegclip.models.backbones.utils import DropPath
from third_party.zegclip.models.decode_heads.atm_head import ATMSingleHeadSeg


def nested_set(dic, key, value):
    keys = key.split('.')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value


def nested_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)


def is_vlm(obj):
    return isinstance(obj, VLM)


def forward_wrapper(self, img, gt=None, need_fp=False, only_fp=False, forward_mode='default'):
    if forward_mode == 'maskclip_trust':
        return self.train_maskclip_trust(img, gt)
    elif forward_mode == 'default':
        x = self.extract_feat(img)
        if self.disable_dropout:
            dropout_modules = [module for module in self.modules() if isinstance(module, torch.nn.Dropout) or isinstance(module, DropPath)]
            for module in dropout_modules:
                module.eval()
        if only_fp:
            if is_vlm(self):
                feats = x[0][0]
                x[0][0] = [F.dropout2d(f, self.fp_rate) for f in feats]
                # perturb features from conv_encoder
                if len(x) == 3 and x[2] is not None:
                    x[2] = [F.dropout2d(f, self.fp_rate) for f in x[2]]
                # also provide unperturbed features
                if hasattr(self.decode_head, 'dc_unperturbed') and self.decode_head.dc_unperturbed:
                    assert len(x[0]) == 2
                    x[0].append(feats)
            else:
                x = [F.dropout2d(f, self.fp_rate) for f in x]
        elif need_fp:
            if is_vlm(self):
                feats = x[0][0]
                x[0][0] = [torch.cat((f, F.dropout2d(f, self.fp_rate))) for f in feats]
                x[0][1] = torch.cat((x[0][1], x[0][1]))
                # perturb features from conv_encoder
                if len(x) == 3 and x[2] is not None:
                    x[2] = [torch.cat((f, F.dropout2d(f, self.fp_rate))) for f in x[2]]
                # also provide unperturbed features
                if hasattr(self.decode_head, 'dc_unperturbed') and self.decode_head.dc_unperturbed:
                    assert len(x[0]) == 2
                    x[0].append([torch.cat((f, f)) for f in feats])
            else:
                x = [torch.cat((f, F.dropout2d(f, self.fp_rate))) for f in x]
        out = self._decode_head_forward_test(x, img_metas=None)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if need_fp:
            out = out.chunk(2)
        return out
    else:
        raise ValueError(forward_mode)

def build_model(cfg):
    model_type = cfg['model']
    if model_type == 'deeplabv3plus':
        model = DeepLabV3Plus(cfg)
    elif 'mmseg.' in model_type:
        model_type = model_type.replace('mmseg.', '')
        model_cfg_file = f'configs/_base_/models/{model_type}.py'
        mmseg_cfg = Config.fromfile(model_cfg_file)
        mmseg_cfg['model']['decode_head']['num_classes'] = cfg['nclass']
        if 'zegclip' in model_type or 'vlm' in model_type:
            if mmseg_cfg['img_size'] != cfg['crop_size']:
                print('Modify model image_size to match crop_size', cfg['crop_size'])
                nested_set(mmseg_cfg, 'img_size', cfg['crop_size'])
                nested_set(mmseg_cfg, 'model.backbone.img_size', (cfg['crop_size'],  cfg['crop_size']))
                nested_set(mmseg_cfg, 'model.decode_head.img_size', cfg['crop_size'])
            emb_dataset_prefix = {
                'pascal': 'voc12_wbg',
                'cityscapes': 'cityscapes',
                'coco': 'coco',
                'ade': 'ade',
            }[cfg['dataset']]
            text_embedding_variant = cfg['text_embedding_variant']
            text_embedding = f'configs/_base_/datasets/text_embedding/{emb_dataset_prefix}_{text_embedding_variant}.npy'
            nested_set(mmseg_cfg, 'model.load_text_embedding', text_embedding)
            mcc_text_embedding_variant = cfg['mcc_text']
            mcc_text_embedding = f'configs/_base_/datasets/text_embedding/{emb_dataset_prefix}_{mcc_text_embedding_variant}.npy'
            nested_set(mmseg_cfg, 'model.load_mcc_text_embedding', mcc_text_embedding)
            pl_text_embedding_variant = cfg['pl_text']
            pl_text_embedding = f'configs/_base_/datasets/text_embedding/{emb_dataset_prefix}_{pl_text_embedding_variant}.npy'
            nested_set(mmseg_cfg, 'model.load_pl_text_embedding', pl_text_embedding)
        if mmseg_cfg['model']['decode_head']['type'] == 'ATMSingleHeadSeg':
            mmseg_cfg['model']['decode_head']['seen_idx'] = list(range(cfg['nclass']))
            mmseg_cfg['model']['decode_head']['all_idx'] = list(range(cfg['nclass']))
        if mmseg_cfg['model']['decode_head'].get('loss_decode') is not None and \
                mmseg_cfg['model']['decode_head']['loss_decode']['type'] == 'SegLossPlus':
            mmseg_cfg['model']['decode_head']['loss_decode']['num_classes'] = cfg['nclass']
        if cfg['clip_encoder'] is not None:
            clip_encoder_cfg = Config.fromfile(f'configs/_base_/models/{cfg["clip_encoder"]}.py')
            clip_encoder_cfg['img_size'] = mmseg_cfg['img_size']
            if cfg.get('mcc_fix_resize_pos'):
                clip_encoder_cfg['backbone']['img_size'] = mmseg_cfg['img_size']
            mmseg_cfg['model']['clip_encoder'] = clip_encoder_cfg['backbone']
        if 'model_args' in cfg:
            mmseg_cfg['model'].update(cfg['model_args'])
        model = build_segmentor(
            mmseg_cfg.model,
            train_cfg=mmseg_cfg.get('train_cfg'),
            test_cfg=mmseg_cfg.get('test_cfg'))
        model.disable_dropout = cfg['disable_dropout']
        model.fp_rate = cfg['fp_rate']
        model.forward = types.MethodType(forward_wrapper, model)
        model.init_weights()
    else:
        raise ValueError(model_type)
    
    return model

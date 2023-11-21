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


import numpy as np
import torch
import torch.nn.functional as F
from mmseg.models import builder
from mmseg.models.builder import SEGMENTORS
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder

from model.text_embeddings import (aggregate_concept_predictions,
                                   get_class_to_concept_idxs)


@SEGMENTORS.register_module()
class VLM(EncoderDecoder):
    def __init__(self,
                 freeze_backbone=False,
                 exclude_keys=None,
                 load_text_embedding=None,
                 load_mcc_text_embedding=None,
                 load_pl_text_embedding=None,
                 clip_encoder=None,
                 conv_encoder=None,
                 maskclip_class_filter=None,
                 maskclip_trust_head=None,
                 renorm_clip_img=False,
                 **args):
        super(VLM, self).__init__(**args)
        assert load_text_embedding == load_pl_text_embedding
        assert maskclip_class_filter is None
        assert maskclip_trust_head is None
        self.local_iter = 0

        self.clip_encoder = None
        if clip_encoder is not None:
            self.clip_encoder = builder.build_backbone(clip_encoder)
        self.conv_encoder = None
        if conv_encoder is not None:
            self.conv_encoder = builder.build_backbone(conv_encoder)

        self.load_text_embedding = load_text_embedding
        self.decode_head.load_text_embedding = load_text_embedding
        self.load_mcc_text_embedding = load_mcc_text_embedding
        self.renorm_clip_img = renorm_clip_img
        if renorm_clip_img:
            print('Renormalize clip image.')
        if self.load_mcc_text_embedding:
            self.loaded_mcc_text_feat = np.load(self.load_mcc_text_embedding)
            self.loaded_mcc_text_feat = torch.from_numpy(self.loaded_mcc_text_feat).float()
        else:
            raise NotImplementedError

        if freeze_backbone:
            self.freeze(self.backbone, exclude_keys=exclude_keys)

    def renormalize_img_for_clip(self, img):
        if not self.renorm_clip_img:
            return img
        loader_mean, loader_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        clip_mean, clip_std = [0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]
        loader_mean = torch.tensor(loader_mean, device=img.device).view(1, -1, 1, 1)
        loader_std = torch.tensor(loader_std, device=img.device).view(1, -1, 1, 1)
        clip_mean = torch.tensor(clip_mean, device=img.device).view(1, -1, 1, 1)
        clip_std = torch.tensor(clip_std, device=img.device).view(1, -1, 1, 1)
        return (img * loader_std + loader_mean - clip_mean) / clip_std

    def freeze(self, model, exclude_keys=None):
        for n, m in model.named_parameters():
            m.requires_grad = False
            if exclude_keys is not None:
                assert isinstance(exclude_keys, list)
                for k in exclude_keys:
                    if str(k) in n:
                        m.requires_grad = True
                        print(f'Finetune {n}')
    
    def forward_maskclip(self, img, conf_tresh):
        img = self.renormalize_img_for_clip(img)
        self.clip_encoder.eval()
        with torch.no_grad():
            text_feat = self.loaded_mcc_text_feat.detach().to(img.device)
            visual_feat, _ = self.clip_encoder(img)
            visual_feat = visual_feat[-1]

            dense_pred = F.conv2d(visual_feat, text_feat[:, :, None, None])
            if dense_pred.shape[1] != self.num_classes:
                cls2con = get_class_to_concept_idxs(self.load_mcc_text_embedding)
                dense_pred = aggregate_concept_predictions(dense_pred, cls2con)
            assert dense_pred.shape[1] == self.num_classes
            dense_pred = F.interpolate(dense_pred, size=img.shape[-2:],
                                       mode='bilinear', align_corners=self.decode_head.align_corners)
            dense_pred = (100.0 * dense_pred).softmax(dim=1)
            dense_pred_certainty, dense_pred = dense_pred.max(dim=1)

            filtered_dense_pred = dense_pred.clone()
            filtered_dense_pred[dense_pred_certainty < conf_tresh] = 255
        return filtered_dense_pred

    def extract_feat(self, img):
        orig_img = img
        img = self.renormalize_img_for_clip(img)
        visual_feat = self.backbone(img)
        text_feat = np.load(self.load_text_embedding)
        text_feat = torch.from_numpy(text_feat).to(img.device)
        self.decode_head.load_text_embedding = self.load_text_embedding
        conv_feat = None
        if self.conv_encoder is not None:
            conv_feat = self.conv_encoder(orig_img)

        return [visual_feat, text_feat, conv_feat]

    def _decode_head_forward_test(self, x, img_metas):
        seg_logits = self.decode_head.forward(x, force_output_pred_masks=True)['pred_masks']
        return seg_logits

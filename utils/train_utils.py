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


def cutmix_img_(img, img_mix, cutmix_box):
    img[cutmix_box.unsqueeze(1).expand(img.shape) == 1] = \
        img_mix[cutmix_box.unsqueeze(1).expand(img.shape) == 1]
  

def cutmix_mask(mask, mask_mix, cutmix_box):
    cutmixed = mask.clone()
    cutmixed[cutmix_box == 1] = mask_mix[cutmix_box == 1]
    return cutmixed


def confidence_weighted_loss(loss, conf_map, ignore_mask, cfg):
    assert loss.dim() == 3
    assert conf_map.dim() == 3
    assert ignore_mask.dim() == 3
    valid_mask = (ignore_mask != 255)
    sum_pixels = dict(dim=(1, 2), keepdim=True)
    if cfg['conf_mode'] == 'pixelwise':
        loss = loss * ((conf_map >= cfg['conf_thresh']) & valid_mask)
        loss = loss.sum() / valid_mask.sum().item()
    elif cfg['conf_mode'] == 'pixelratio':
        ratio_high_conf = ((conf_map >= cfg['conf_thresh']) & valid_mask).sum(**sum_pixels) / valid_mask.sum(**sum_pixels)
        loss = loss * ratio_high_conf
        loss = loss.sum() / valid_mask.sum().item()
    elif cfg['conf_mode'] == 'pixelavg':
        avg_conf = (conf_map * valid_mask).sum(**sum_pixels) / valid_mask.sum(**sum_pixels)
        loss = loss.sum() * avg_conf
        loss = loss.sum() / valid_mask.sum().item()
    else:
        raise ValueError(cfg['conf_mode'])
    return loss


class DictAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avgs = {}
        self.sums = {}
        self.counts = {}

    def update(self, vals):
        for k, v in vals.items():
            if torch.is_tensor(v):
                v = v.detach()
            if k not in self.sums:
                self.sums[k] = 0
                self.counts[k] = 0
            self.sums[k] += v
            self.counts[k] += 1
            self.avgs[k] = torch.true_divide(self.sums[k], self.counts[k])

    def __str__(self):
        s = []
        for k, v in self.avgs.items():
            s.append(f'{k}: {v:.3f}')
        return ', '.join(s)

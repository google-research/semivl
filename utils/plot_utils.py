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
from matplotlib import pyplot as plt


def colorize_label(seg, palette):
    color_seg = 255 * np.ones((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(palette):
        if not np.all(color == [255, 255, 255]):
            color_seg[seg == label, :] = color
    return color_seg


def plot_data(ax, title, type, data, palette=None):
    data = data.cpu()
    if type == 'image':
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        data = data.permute([1, 2, 0]).mul(std).add(mean)
        ax.imshow(data)
    elif type == 'label':
        out = colorize_label(data.squeeze(0), palette)
        ax.imshow(out)
    elif type == 'prediction':
        data = data.squeeze(0).argmax(dim=0)
        out = colorize_label(data, palette)
        ax.imshow(out)
    elif type == 'heatmap':
        if data.shape[0] == 1:
            data = data.squeeze(0)
        ax.imshow(data)
    if title is not None:
        ax.set_title(title)
    ax.axis('off')
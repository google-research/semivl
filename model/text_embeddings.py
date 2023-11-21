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
import os
import numpy as np
import clip
import argparse


## VOC12 w/ background
VOC12_wbg_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']


VOC12_wbg_classes_w_concepts4 = [
        ['background', 'bed', 'building', 'cabinet', 'ceiling', 'curtain', 'door', 'fence',
         'floor', 'grass', 'ground', 'mountain', 'road', 'rock', 'shelves', 'sidewalk', 'sky',
         'snow', 'tree', 'wall', 'water', 'window', 'hang glider', 'helicopter', 'jet ski',
         'go-cart', 'tractor', 'emergency vehicle', 'lorry', 'truck', 'lion', 'stool', 'bench',
         'wheelchair', 'coffee table', 'desk', 'side table', 'picnic bench', 'wolve',
         'flowers in a vase', 'goat', 'tram', 'laptop', 'advertising display', 'vehicle interior'],
        ['aeroplane', 'airplane', 'glider'],
        ['bicycle', 'tricycle', 'unicycle'],
        ['bird'],
        ['boat', 'ship', 'rowing boat', 'pedalo'],
        ['bottle', 'plastic bottle', 'glass bottle', 'feeding bottle'],
        ['bus', 'minibus'],
        ['car', 'van', 'large family car', 'realistic toy car'],
        ['cat', 'domestic cat'],
        ['chair', 'armchair', 'deckchair'],
        ['cow'],
        ['dining table', 'table for eating at'],
        ['dog', 'domestic dog'],
        ['horse', 'pony', 'donkey', 'mule'],
        ['motorbike', 'moped', 'scooter', 'sidecar'],
        ['person', 'people', 'baby', 'face'],
        ['potted plant', 'indoor plant in a pot', 'outdoor plant in a pot'],
        ['sheep'],
        ['sofa'],
        ['train', 'train carriage'],
        ['tv', 'monitor', 'standalone screen'],
]


## COCO Stuff
COCO_stuff_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner',
        'blanket', 'branch', 'bridge', 'building-other', 'bush', 'cabinet',
        'cage', 'cardboard', 'carpet', 'ceiling-other', 'ceiling-tile',
        'cloth', 'clothes', 'clouds', 'counter', 'cupboard', 'curtain',
        'desk-stuff', 'dirt', 'door-stuff', 'fence', 'floor-marble',
        'floor-other', 'floor-stone', 'floor-tile', 'floor-wood',
        'flower', 'fog', 'food-other', 'fruit', 'furniture-other', 'grass',
        'gravel', 'ground-other', 'hill', 'house', 'leaves', 'light', 'mat',
        'metal', 'mirror-stuff', 'moss', 'mountain', 'mud', 'napkin', 'net',
        'paper', 'pavement', 'pillow', 'plant-other', 'plastic', 'platform',
        'playingfield', 'railing', 'railroad', 'river', 'road', 'rock', 'roof',
        'rug', 'salad', 'sand', 'sea', 'shelf', 'sky-other', 'skyscraper',
        'snow', 'solid-other', 'stairs', 'stone', 'straw', 'structural-other',
        'table', 'tent', 'textile-other', 'towel', 'tree', 'vegetable',
        'wall-brick', 'wall-concrete', 'wall-other', 'wall-panel',
        'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'waterdrops',
        'window-blind', 'window-other', 'wood']

# COCO classes
COCO_classes = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
        'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
        'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
        'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# ADE20k classes
ADE_classes = [
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed ", "windowpane", "grass", "cabinet",
    "sidewalk", "person", "earth", "door", "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field", "armchair", "seat", "fence", "desk",
    "rock", "wardrobe", "lamp", "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest of drawers","counter", "sand", "sink", "skyscraper", "fireplace", "refrigerator", "grandstand",
    "path", "stairs", "runway", "case", "pool table", "pillow", "screen door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee table", "toilet", "flower", "book", "hill", "bench", "countertop", "stove",
    "palm", "kitchen island", "computer", "swivel chair", "boat", "bar", "arcade machine", "hovel", "bus",
    "towel", "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth", "television receiver",
    "airplane", "dirt track", "apparel", "pole", "land", "bannister", "escalator", "ottoman", "bottle",
    "buffet", "poster", "stage", "van", "ship", "fountain", "conveyer belt", "canopy", "washer", "plaything",
    "swimming pool", "stool", "barrel", "basket", "waterfall", "tent", "bag", "minibike", "cradle", "oven",
    "ball", "food", "step", "tank", "trade name", "microwave", "pot", "animal", "bicycle", "lake", "dishwasher",
    "screen", "blanket", "sculpture", "hood", "sconce", "vase", "traffic light", "tray", "ashcan", "fan", "pier",
    "crt screen", "plate", "monitor", "bulletin board", "shower", "radiator", "glass", "clock", "flag"]

## Cityscapes
Cityscapes_classes = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
        'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
        'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
        'bicycle']



Cityscapes_classes_w_concepts3 = [
    ['road', 'street', 'parking space'],
    ['sidewalk'],
    ['building', 'skyscaper', 'house', 'bus stop building', 'garage', 'car port', 'scaffolding'],
    ['individual standing wall, which is not part of a building'],
    ['fence', 'hole in fence'],
    ['pole', 'sign pole', 'traffic light pole'],
    ['traffic light'],
    ['traffic sign', 'parking sign', 'direction sign'],
    ['vegetation', 'tree', 'hedge'],
    ['terrain', 'grass', 'soil', 'sand'],
    ['sky'],
    ['person', 'pedestrian', 'walking person', 'standing person', 'person sitting on the ground',
     'person sitting on a bench', 'person sitting on a chair'],
    ['rider', 'cyclist', 'motorcyclist'],
    ['car', 'jeep', 'SUV', 'van'],
    ['truck', 'box truck', 'pickup truck', 'truck trailer'],
    ['bus'],
    ['train', 'tram'],
    ['motorcycle', 'moped', 'scooter'],
    ['bicycle'],
]


def single_template(save_path, class_names, model):
    with torch.no_grad():
        texts = torch.cat([clip.tokenize(f"a photo of a {c}") for c in class_names]).cuda()
        text_embeddings = model.encode_text(texts)
        text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
        np.save(save_path, text_embeddings.cpu().numpy())
    return text_embeddings

def single_template_concept_avg(save_path, class_concept_list, model):
    all_concepts = [j for sub in class_concept_list for j in sub]
    print(all_concepts)
    with torch.no_grad():
        tokens = torch.cat([clip.tokenize(f"a photo of a {c}") for c in all_concepts]).cuda()
        concept_embeddings = model.encode_text(tokens)
        print(concept_embeddings.shape)
        avg_concept_embeddings = []
        concept_idx = 0
        for i, concepts in enumerate(class_concept_list):
            n_concepts_for_class = len(concepts)
            print(concept_idx, concept_idx + n_concepts_for_class, all_concepts[concept_idx : concept_idx+n_concepts_for_class])
            concepts_for_class = concept_embeddings[concept_idx : concept_idx+n_concepts_for_class]
            assert concepts_for_class.shape[0] == n_concepts_for_class
            avg_concept = torch.sum(concepts_for_class, dim=0) / n_concepts_for_class
            avg_concept_embeddings.append(avg_concept)
            concept_idx += n_concepts_for_class
        avg_concept_embeddings = torch.stack(avg_concept_embeddings, dim=0)
        assert avg_concept_embeddings.shape[0] == len(class_concept_list)
        avg_concept_embeddings = avg_concept_embeddings / avg_concept_embeddings.norm(dim=-1, keepdim=True)
        if save_path is not None:
            np.save(save_path, avg_concept_embeddings.cpu().numpy())
    return avg_concept_embeddings

def aggregate_concept_predictions(pred, class_to_concept_idxs):
    B, _, H, W = pred.shape
    agg_pred = torch.zeros(B, len(class_to_concept_idxs), H, W, device=pred.device)
    for cls_i, conc_i in class_to_concept_idxs.items():
        agg_pred[:, cls_i] = pred[:, conc_i].max(dim=1).values
    return agg_pred

def flatten_class_concepts(class_concepts):
    concepts = []
    concept_to_class_idx = {}
    class_to_concept_idxs = {}
    for i, cls_concepts in enumerate(class_concepts):
        for concept in cls_concepts:
            concept_to_class_idx[len(concepts)] = i
            if i not in class_to_concept_idxs:
                class_to_concept_idxs[i] = []
            class_to_concept_idxs[i].append(len(concepts))
            concepts.append(concept)
    return concepts, concept_to_class_idx, class_to_concept_idxs

def get_class_to_concept_idxs(save_path):
    if save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept4_single.npy':
        _, _, class_to_concept_idxs = flatten_class_concepts(VOC12_wbg_classes_w_concepts4)
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_concept3_single.npy':
        _, _, class_to_concept_idxs = flatten_class_concepts(Cityscapes_classes_w_concepts3)
    else:
        raise ValueError(save_path)
    return class_to_concept_idxs

def prepare_text_embedding(save_path):
    assert os.path.isdir(os.path.dirname(save_path))
    # assert not os.path.isfile(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load('ViT-B/16', device)

    ## VOC12 with background:
    if save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_single.npy':
        single_template(save_path, VOC12_wbg_classes, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_conceptavg4_single.npy':
        single_template_concept_avg(save_path, VOC12_wbg_classes_w_concepts4, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept4_single.npy':
        flat_concepts, _, _ = flatten_class_concepts(VOC12_wbg_classes_w_concepts4)
        single_template(save_path, flat_concepts, model)
    ## COCO:
    elif save_path == 'configs/_base_/datasets/text_embedding/coco_single.npy':
        single_template(save_path, COCO_classes, model)
    ## Cityscapes:
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_single.npy':
        single_template(save_path, Cityscapes_classes, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_concept3_single.npy':
        flat_concepts, _, _ = flatten_class_concepts(Cityscapes_classes_w_concepts3)
        single_template(save_path, flat_concepts, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_conceptavg3_single.npy':
        single_template_concept_avg(save_path, Cityscapes_classes_w_concepts3, model)
    ## ADE20k:
    elif save_path == 'configs/_base_/datasets/text_embedding/ade_single.npy':
        single_template(save_path, ADE_classes, model)
    else:
        raise NotImplementedError(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()
    prepare_text_embedding(f'configs/_base_/datasets/text_embedding/{args.name}.npy')

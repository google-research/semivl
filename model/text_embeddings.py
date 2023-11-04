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

## VOC12
VOC12_classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

## VOC12 w/ background
VOC12_wbg_classes = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']

VOC12_wbg_classes_w_concepts = [
        ['background', 'hang glider', 'helicopter', 'jet ski', 'go-cart', 'tractor', 'emergency vehicle', 'lorry', 'truck', 'lion', 'stool', 'bench', 'wheelchair', 'coffee table', 'desk', 'side table', 'picnic bench', 'wolve', 'flowers in a vase', 'goat', 'sofabed', 'tram', 'laptop', 'advertising display'],
        ['aeroplane', 'glider'],
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

VOC12_wbg_classes_w_concepts2 = [
        ['background'],
        ['aeroplane', 'glider'],
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

VOC12_wbg_classes_w_concepts3 = [
        ['background', 'bed', 'building', 'cabinet', 'ceiling', 'curtain', 'door', 'fence', 'floor', 'grass', 'ground', 'mountain', 'road', 'rock', 'shelves', 'sidewalk', 'sky', 'snow', 'tree', 'wall', 'water', 'window', 'hang glider', 'helicopter', 'jet ski', 'go-cart', 'tractor', 'emergency vehicle', 'lorry', 'truck', 'lion', 'stool', 'bench', 'wheelchair', 'coffee table', 'desk', 'side table', 'picnic bench', 'wolve', 'flowers in a vase', 'goat', 'tram', 'laptop', 'advertising display'],
        ['aeroplane', 'glider'],
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
        ['potted plant', 'plant in a pot'],
        ['sheep'],
        ['sofa'],
        ['train', 'train carriage'],
        ['tv', 'monitor', 'standalone screen'],
]

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

VOC12_wbg_classes_w_definition = [
        ['background', 'building', 'ground', 'grass', 'tree', 'sky'],
        ['airplane', 'a flying vehicle with fixed wings'],
        ['bicycle', 'a vehicle consisting of two wheels held in a frame one behind the other, propelled by pedals and steered with handlebars attached to the front wheel'],
        ['bird', 'a warm-blooded egg-laying vertebrate animal distinguished by the possession of feathers, wings, a beak, and typically by being able to fly'],
        ['boat', 'a vessel for travelling over water, propelled by oars, sails, or an engine'],
        ['bottle', 'a glass or plastic container with a narrow neck, used for storing drinks or other liquids'],
        ['bus', 'a large motor vehicle carrying passengers by road'],
        ['car', 'a four-wheeled road vehicle that is powered by an engine and is able to carry a small number of people'],
        ['cat', 'a small domesticated carnivorous mammal with soft fur, a short snout, and retractable claws'],
        ['chair', 'a separate seat for one person, typically with a back and four legs'],
        ['cow', 'a fully grown female animal of a domesticated breed of ox, kept to produce milk or beef'],
        ['dining table', 'a table on which meals are served in a dining room'],
        ['dog', 'a domesticated carnivorous mammal that typically has a long snout and non-retractable claws'],
        ['horse', 'a large plant-eating domesticated mammal with solid hoofs and a flowing mane and tail, used for riding, racing, and to carry and pull loads'],
        ['motorbike', 'a two-wheeled vehicle that is powered by a motor and has no pedals'],
        ['person', 'a human being regarded as an individual'],
        ['potted plant', 'a plant in a pot'],
        ['sheep', 'a domesticated ruminant mammal with a thick woolly coat'],
        ['sofa', 'a long upholstered seat with a back and arms, for two or more people'],
        ['train', 'a series of connected railway carriages or wagons moved by a locomotive or by integral motors'],
        ['tv monitor', 'a flat panel or area on an electronic device such as a television, computer, or smartphone, on which images and data are displayed'],
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

Cityscapes_classes_w_concepts2 = [
    # Part of ground on which cars usually drive, i.e. all lanes, all directions, all streets.
    # Including the markings on the road. Areas only delimited by markings from the main road (no
    # texture change) are also road, e.g. bicycle lanes, roundabout lanes, or parking spaces.
    # This label does not include curbs.
    ['road', 'lane', 'street', 'road marking', 'bicycle lane', 'roundabout lane', 'parking space', 'street rails'],
    # Part of ground designated for pedestrians or cyclists. Delimited from the road by some
    # obstacle, e.g. curbs or poles (might be small), not only by markings. Often elevated compared
    # to the road. Often located at the sides of a road. This label includes a possibly delimiting
    # curb, traffic islands (the walkable part), or pedestrian zones (where usually cars are not
    # allowed to drive during day-time).
    ['sidewalk', 'curb next to sidewalk', 'walkable part of traffic island'],
    # Building, skyscraper, house, bus stop building, garage, car port. If a building has a glass
    # wall that you can see through, the wall is still building. Includes scaffolding attached to
    # buildings.
    ['building', 'skyscaper', 'house', 'bus stop building', 'garage', 'car port', 'building wall',
     'scaffolding', 'commercial sign attached to building', 'advertisement sign attached to building',
     'plant attached to building'],
    # Individual standing wall. Not part of a building.
    ['individual standing wall, which is not part of a building'],
    # Fence including any holes.
    ['fence', 'hole in fence'],
    # Small mainly vertically oriented pole. E.g. sign pole, traffic light poles. If the pole
    # has a horizontal part (often for traffic light poles) this part is also considered pole.
    # If there are things mounted at the pole that are neither traffic light nor traffic sign
    # (e.g. street lights) and that have a diameter (in pixels) of at most twice the diameter
    # of the pole, then these things might also be labeled pole. If they are larger, they are
    # labeled static.
    ['pole', 'sign pole', 'traffic light pole', 'street light'],
    # The traffic light box without its poles.
    ['traffic light'],
    # Sign installed from the state/city authority, usually for information of the driver/
    # cyclist/pedestrian in an everyday traffic scene, e.g. traffic- signs, parking signs,
    # direction signs - without their poles. No ads/commercial signs. Only the front side of a
    # sign containing the information. The back side is static. Note that commercial signs
    # attached to buildings become building, attached to poles or standing on their own become
    # static.
    ['traffic sign', 'sign installed from the city authority', 'parking sign', 'direction sign'],
    # Tree, hedge, all kinds of vertical vegetation. Plants attached to buildings are usually
    # not annotated separately and labeled building as well. If growing at the side of a wall
    # or building, marked as vegetation if it covers a substantial part of the surface (more
    # than 20%).
    ['vegetation', 'tree', 'hedge', 'plant'],
    # Grass, all kinds of horizontal vegetation, soil or sand. These areas are not meant to
    # be driven on. This label includes a possibly delimiting curb. Single grass stalks do
    # not need to be annotated and get the label of the region they are growing on.
    ['terrain', 'grass', 'soil', 'sand', 'curb next to terrain', 'rail track'],
    # Open sky, without leaves of tree. Includes thin electrical wires in front of the sky.
    ['sky', 'overhead line'],
    # A human that satisfies the following criterion. Assume the human moved a distance of 1m and
    # stopped again. If the human would walk, the label is person, otherwise not. Examples are
    # people walking, standing or sitting on the ground, on a bench, on a chair. This class also
    # includes toddlers, someone pushing a bicycle or standing next to it with both legs on the
    # same side of the bicycle. This class includes anything that is carried by the person, e.g.
    # backpack, but not items touching the ground, e.g. trolleys.
    ['pedestrian', 'walking person', 'standing person', 'person sitting on the ground',
     'person sitting on a bench', 'person sitting on a chair', 'toddler', 'person pushing a bicycle', 
     'person standing next to bicycle',
     'item carried by person'],
    # A human that would use some device to move a distance of 1m. Includes, riders/drivers of
    # bicycle, motorbike, scooter, skateboards, horses, roller-blades, wheel-chairs, road cleaning
    # cars, cars without roof. Note that a visible driver of a car with roof can only be seen
    # through the window. Since holes are not labeled, the human is included in the car label.
    ['rider', 'cyclist', 'motorcyclist', 'scooter rider', 'skateboarder', 'horse rider',
     'person with roller-blades', 'wheelchair user', 'driver of a car without roof'],
    # Car, jeep, SUV, van with continuous body shape, caravan, no other trailers.
    ['car', 'jeep', 'SUV', 'van', 'caravan'],
    # Truck, box truck, pickup truck. Including their trailers. Back part / loading area is
    # physically separated from driving compartment
    # ['truck', 'box truck', 'pickup truck', 'truck trailer'],
    ['truck with separate driving compartment', 'box truck with separate driving compartment', 'pickup truck with separate driving compartment', 'truck trailer'],
    # Bus for 9+ persons, public transport or long distance transport.
    ['public transport bus', 'long distance bus'],
    # Vehicle on rails, e.g. tram, train.
    ['train', 'tram'],
    # Motorbike, moped, scooter without the driver (that's a rider, see above).
    ['motorcycle', 'moped', 'scooter'],
    # Bicycle without the driver (that's a rider, see above).
    ['bicycle'],
]


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
    if save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept2_single.npy':
        _, _, class_to_concept_idxs = flatten_class_concepts(VOC12_wbg_classes_w_concepts2)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept3_single.npy':
        _, _, class_to_concept_idxs = flatten_class_concepts(VOC12_wbg_classes_w_concepts3)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept4_single.npy':
        _, _, class_to_concept_idxs = flatten_class_concepts(VOC12_wbg_classes_w_concepts4)
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_concept2_single.npy':
        _, _, class_to_concept_idxs = flatten_class_concepts(Cityscapes_classes_w_concepts2)
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
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_def_single.npy':
        single_template_concept_avg(save_path, VOC12_wbg_classes_w_definition, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_conceptavg_single.npy':
        single_template_concept_avg(save_path, VOC12_wbg_classes_w_concepts, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_conceptavg2_single.npy':
        single_template_concept_avg(save_path, VOC12_wbg_classes_w_concepts2, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_conceptavg4_single.npy':
        single_template_concept_avg(save_path, VOC12_wbg_classes_w_concepts4, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept2_single.npy':
        flat_concepts, _, _ = flatten_class_concepts(VOC12_wbg_classes_w_concepts2)
        single_template(save_path, flat_concepts, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept3_single.npy':
        flat_concepts, _, _ = flatten_class_concepts(VOC12_wbg_classes_w_concepts3)
        single_template(save_path, flat_concepts, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_wbg_concept4_single.npy':
        flat_concepts, _, _ = flatten_class_concepts(VOC12_wbg_classes_w_concepts4)
        single_template(save_path, flat_concepts, model)
    ## VOC12:
    elif save_path == 'configs/_base_/datasets/text_embedding/voc12_single.npy':
        single_template(save_path, VOC12_classes, model)
    ## COCO:
    elif save_path == 'configs/_base_/datasets/text_embedding/coco_single.npy':
        single_template(save_path, COCO_classes, model)
    ## Cityscapes:
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_single.npy':
        single_template(save_path, Cityscapes_classes, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_concept2_single.npy':
        flat_concepts, _, _ = flatten_class_concepts(Cityscapes_classes_w_concepts2)
        single_template(save_path, flat_concepts, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_concept3_single.npy':
        flat_concepts, _, _ = flatten_class_concepts(Cityscapes_classes_w_concepts3)
        single_template(save_path, flat_concepts, model)
    elif save_path == 'configs/_base_/datasets/text_embedding/cityscapes_conceptavg2_single.npy':
        single_template_concept_avg(save_path, Cityscapes_classes_w_concepts2, model)
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

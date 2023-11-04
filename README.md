# SemiVL: Semi-Supervised Semantic Segmentation with Vision-LanguageGuidance

TODO: Authors

TODO: Arxiv Link

## Overview

In semi-supervised semantic segmentation, a model is trained with a limited number of labeled images along with a large corpus of unlabeled images to reduce the high annotation effort. While previous methods are able to learn good segmentation boundaries, they often struggle with capturing all relevant semantics due to the limited supervision. We propose SemiVL, a framework to supplement the semi-supervised training with vision-language guidance to learn better fine-grained semantic discriminability. It comprises of five strategies:

1) We initialize the semi-supervised training with a Vision-Language Model (VLM) pre-trained on image-caption pairs to utilize their rich semantics.
2) We fine-tune only the attention layers to effectively adapt the VLM from image-level to dense predictions while retaining its rich semantics.
3) We introduce a language-guided decoder to utilize the VLM's capability to jointly reason over vision and language.
4) We introduce dense CLIP guidance to generate additional pseudo-labels for self-training. We find this helps anchoring the self-training to better capture semantics.
5) We utilize the new possibility of SemiVL to provide language guidance in the form of class definitions to the model.

We evaluate SemiVL on four common semantic segmentation datasets (Pascal VOC12, COCO, ADE20K, and Cityscapes), where it significantly outperforms previous semi-supervised methods. SemiVL significantly improves the state-of-the-art by +15.3 mIoU on COCO with 232 annotations, by +7.9 mIoU on ADE20K with 158 annotations, or +5.8 mIoU on VOC12 with 92 annotations. In particular, our approach can maintain the performance of previous methods using only a quarter of the segmentation annotations.

TODO: Overview figure

TODO: Bibtex

## Getting Started

### Environment

Create a conda environment:

```bash
conda create -n semivl python=3.7.13
conda activate semivl
```

Install the required pip packages:

```bash
pip install -r requirements.txt
```

Install mmcv and mmsegmentation:

```bash
mim install mmcv-full==1.4.4
pip install mmsegmentation==0.24.0
```

### Pre-Trained Backbones

Download and convert the CLIP ViT-B/16 weights:

```bash
mkdir -p pretrained/
python third_party/maskclip/convert_clip_weights.py --model ViT16 --backbone
```

If you want to also want to run the UniMatch baseline, follow [UniMatch#pretrained-backbone](https://github.com/LiheYoung/UniMatch#pretrained-backbone).

### Datasets


Please, download the following datasets:

- Pascal VOC12: [JPEGImages](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) | [SegmentationClass](https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing)
- Cityscapes: [leftImg8bit](https://www.cityscapes-dataset.com/file-handling/?packageID=3) | [gtFine](https://drive.google.com/file/d/1E_27g9tuHm6baBqcA7jct_jqcGA89QPm/view?usp=sharing)
- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip) | [val2017](http://images.cocodataset.org/zips/val2017.zip) | [masks](https://drive.google.com/file/d/166xLerzEEIbU7Mt1UGut-3-VN41FMUb1/view?usp=sharing)
- ADE20K: [ADEChallengeData2016](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)

Please, extract the datasets to the following folder structure:

```
$HOME/data/
├── voc
    ├── JPEGImages
    └── SegmentationClass
├── cityscapes
    ├── leftImg8bit
    └── gtFine
├── coco
    ├── train2017
    ├── val2017
    └── masks
├── ADEChallengeData2016
    ├── images
    └── annotations
```

Note: The ground truth of VOC12, Cityscapes, and COCO have been preprocessed by [UniMatch](https://github.com/LiheYoung/UniMatch).

TODO: Provide preprocessing scripts

## Training

To launch a training job, please run:

```bash
python experiments.py --exp EXP_ID --run RUN_ID
# e.g. EXP_ID=40; RUN_ID=0 for SemiVL on VOC with 92 labels
```

It will automatically generate the relevant config files in `configs/generated/` and start the corresponding training job.

For more information on the available experiments and runs, please refer to `def generate_experiment_cfgs(exp_id)` in [experiments.py](experiments.py).

The training log, tensorboard, checkpoints, and debug images are stored in `exp/`.

## Results and Checkpoints

TODO

## Framework Structure

TODO

## Acknowledgements

SemiVL is based on [UniMatch](https://github.com/LiheYoung/UniMatch), [MaskCLIP](https://github.com/chongzhou96/MaskCLIP), [ZegCLIP](https://github.com/ZiqinZhou66/ZegCLIP), and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation). We thank their authors for making the source code publicly available.

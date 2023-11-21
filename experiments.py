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

from argparse import ArgumentParser
from functools import reduce
import itertools
import yaml
import os
import os.path as osp
import subprocess
import collections.abc
from version import __version__


DATA_DIR = '~/data/'


def nested_set(dic, key, value):
    keys = key.split('.')
    for key in keys[:-1]:
        dic = dic.setdefault(key, {})
    dic[keys[-1]] = value

def nested_get(dictionary, keys, default=None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split("."), dictionary)

def nested_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = nested_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_git_revision() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except subprocess.CalledProcessError:
        return ''

def human_format(num):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

def config_from_vars(
    exp_id,
    gpu_model='a100',
    n_gpus=4,
    n_nodes=1,
    batch_size=2,
    epochs=80,
    iters=None,
    scheduler_max_iters=None,
    dataset='pascal',
    split='92',
    img_scale=[2048, 512],
    scale_ratio_range=(0.5, 2.0),
    crop_size=512,
    labeled_photometric_distortion=False,
    renorm_clip_img=False,
    method='semivl',
    use_fp=True,
    conf_mode='pixelwise',
    conf_thresh=0.95,
    pleval=True,
    disable_dropout=True,
    fp_rate=0.5,
    maskclip_consistency_lambda=0,
    maskclip_class_filter=None,
    mcc_conf_thresh=0.75,
    mcc_loss_reduce='mean',
    mcc_text='same',
    mcc_fix_resize_pos=False,
    pl_text='same',
    opt='adamw',
    lr=1e-4,
    backbone_lr_mult=10.0,
    conv_enc_lr_mult=1.0,
    warmup_iters=0,
    criterion='mmseg',
    criterion_u='mmseg',
    model='mmseg.zegclip-vitb',
    text_embedding_variant='single',
    eval_mode='zegclip_sliding_window',
    eval_every=1,
    nccl_p2p_disable=False,
):
    cfg = dict()
    name = ''

    # Dataset
    cfg['dataset'] = dataset
    name += dataset.replace('pascal', 'voc').replace('cityscapes', 'cs')
    cfg['data_root'] = dict(
        pascal=osp.join(DATA_DIR, 'voc/'),
        cityscapes=osp.join(DATA_DIR, 'cityscapes/'),
        coco=osp.join(DATA_DIR, 'coco/'),
        ade=osp.join(DATA_DIR, 'ADEChallengeData2016/'),
    )[dataset]
    cfg['nclass'] = dict(
        pascal=21,
        cityscapes=19,
        coco=81,
        ade=150,
    )[dataset]
    if dataset == 'ade':
        cfg['reduce_zero_label'] = True
    cfg['split'] = split
    name += f'-{split}'
    cfg['img_scale'] = img_scale
    if img_scale is not None:
        name += f'-{img_scale}'
    cfg['scale_ratio_range'] = scale_ratio_range
    if scale_ratio_range != (0.5, 2.0):
        name += f'-s{scale_ratio_range[0]}-{scale_ratio_range[1]}'
    cfg['crop_size'] = crop_size
    name += f'-{crop_size}'
    cfg['labeled_photometric_distortion'] = labeled_photometric_distortion
    if labeled_photometric_distortion:
        name += '-phd'

    # Model
    name += f'_{model}'.replace('mmseg.', '').replace('zegclip', 'zcl')
    cfg['model_args'] = {}
    if model == 'dlv3p-r101':
        cfg['model'] = 'deeplabv3plus'
        cfg['backbone'] = 'resnet101'
        cfg['replace_stride_with_dilation'] = [False, False, True]
        cfg['dilations'] = [6, 12, 18]
    elif model == 'dlv3p-xc65':
        cfg['model'] = 'deeplabv3plus'
        cfg['backbone'] = 'xception'
        cfg['dilations'] = [6, 12, 18]
    else:
        cfg['model'] = model
        cfg['text_embedding_variant'] = text_embedding_variant
        cfg['mcc_text'] = text_embedding_variant if mcc_text == 'same' else mcc_text
        cfg['pl_text'] = text_embedding_variant if pl_text == 'same' else pl_text
        text_variant_abbrev = {
            'conceptavg_single': 'cavgs',
            'conceptavg2_single': 'cavg2s',
            'conceptavg3_single': 'cavg3s',
            'conceptavg4_single': 'cavg4s',
            'concept2_single': 'c2s',
            'concept3_single': 'c3s',
            'concept4_single': 'c4s',
            'multi': 'm',
        }
        if text_embedding_variant != 'single':
            name += '-t' + text_variant_abbrev[text_embedding_variant]
        if mcc_text != 'same':
            name += '-mt' + text_variant_abbrev[mcc_text]
        if pl_text != 'same':
            name += '-pt' + text_variant_abbrev[pl_text]


    # Method
    cfg['method'] = method
    name += f'_{method}'.replace('semivl', 'svl').replace('unimatch', 'um').replace('supervised', 'sup')
    if method in ['unimatch', 'semivl']:
        cfg['use_fp'] = use_fp
        if not use_fp:
            name += '-nfp'
        cfg['conf_mode'] = conf_mode
        name += {
            'pixelwise': '',
            'pixelratio': '-cpr',
            'pixelavg': '-cpa',
        }[conf_mode]
        cfg['conf_thresh'] = conf_thresh
        name += f'-{conf_thresh}'
    cfg['disable_dropout'] = disable_dropout
    if disable_dropout:
        name += '-disdrop'
    if method in ['unimatch', 'semivl']:
        cfg['pleval'] = pleval
        if pleval:
            name += '-plev'
    cfg['fp_rate'] = fp_rate
    if fp_rate != 0.5:
        name += f'-fpr{fp_rate}'
    cfg['maskclip_consistency_lambda'] = maskclip_consistency_lambda
    if maskclip_consistency_lambda != 0:
        cfg['clip_encoder'] = 'mcvit16'
        name += f'-mcc{maskclip_consistency_lambda}'
    else:
        cfg['clip_encoder'] = None
    cfg['mcc_conf_thresh'] = mcc_conf_thresh
    if mcc_conf_thresh != 0.75:
        name += f'c{mcc_conf_thresh}'
    cfg['mcc_loss_reduce'] = mcc_loss_reduce
    name += {
        'mean': '',
        'mean_valid': '-mv',
        'mean_all': '-ma',
    }[mcc_loss_reduce]
    cfg['model_args']['maskclip_class_filter'] = {
        None: None,
        1: [9, 18],  # chair and sofa
        2: list(range(1, 21)),  # no background
    }[maskclip_class_filter]
    if maskclip_class_filter is not None:
        name += f'-cf{maskclip_class_filter}'
    if renorm_clip_img:
        cfg['model_args']['renorm_clip_img'] = True
        name += '-rnci'
    if mcc_fix_resize_pos and cfg['clip_encoder'] is not None and crop_size != 512:
        cfg['mcc_fix_resize_pos'] = True
        name += '-frp'

    # Criterion
    cfg['criterion'] = dict(
        name=criterion,
        kwargs=dict(ignore_index=255)
    )
    if cfg['criterion'] == 'OHEM':
        cfg['criterion']['kwargs'].update(dict(
            thresh=0.7,
            min_kept=200000
        ))
    if criterion != 'mmseg':
        name += f'-{criterion}'.replace('CELoss', 'ce').replace('OHEM', 'oh')
    cfg['criterion_u'] = criterion_u
    if criterion_u != 'mmseg':
        name += f'-u{criterion_u}'.replace('CELoss', 'ce')

    # Optimizer
    if opt == 'original':
        cfg['lr'] = lr
        cfg['lr_multi'] = 10.0 if dataset != 'cityscapes' else 1.0
    elif opt == 'adamw':
        cfg['optimizer'] = dict(
            type='AdamW', lr=lr, weight_decay=0.01,
            paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=backbone_lr_mult),
                                            'text_encoder': dict(lr_mult=0.0),
                                            'conv_encoder': dict(lr_mult=conv_enc_lr_mult),
                                            'norm': dict(decay_mult=0.),
                                            'ln': dict(decay_mult=0.),
                                            'head': dict(lr_mult=10.),
                                            }))
    else:
        raise NotImplementedError(opt)
    name += f'_{opt}-{lr:.0e}'.replace('original', 'org')
    if backbone_lr_mult != 10.0:
        name += f'-b{backbone_lr_mult}'
    if conv_enc_lr_mult != 1.0:
        name += f'-cl{conv_enc_lr_mult}'
    cfg['warmup_iters'] = warmup_iters
    cfg['warmup_ratio'] = 1e-6
    if warmup_iters > 0:
        name += f'-w{human_format(warmup_iters)}'

    # Batch
    cfg['gpu_model'] = gpu_model
    cfg['n_gpus'] = n_gpus
    cfg['n_nodes'] = n_nodes
    cfg['batch_size'] = batch_size
    if n_gpus != 4 or batch_size != 2 or n_nodes != 1:
        name += f'_{n_nodes}x{n_gpus}x{batch_size}'

    # Schedule
    assert not (iters is not None and epochs is not None)
    cfg['epochs'] = epochs
    cfg['iters'] = iters
    if epochs is not None and epochs != 80:
        name += f'-ep{human_format(epochs)}'
    if iters is not None:
        name += f'-i{human_format(iters)}'
    if scheduler_max_iters is not None:
        cfg['scheduler_max_iters'] = scheduler_max_iters
        name += f'-smi{scheduler_max_iters}'

    # Eval
    cfg['eval_mode'] = eval_mode
    if eval_mode == 'zegclip_sliding_window':
        cfg['stride'] = 426
    name += '_e' + {
        'original': 'or',
        'sliding_window': 'sw',
        'zegclip_sliding_window': 'zsw',
    }[eval_mode]
    cfg['eval_every_n_epochs'] = eval_every
    cfg['nccl_p2p_disable'] = nccl_p2p_disable


    cfg['exp'] = exp_id
    cfg['name'] = name.replace('.0_', '').replace('.0-', '').replace('.', '').replace('True', 'T')\
        .replace('False', 'F').replace('None', 'N').replace('[', '')\
        .replace(']', '').replace('(', '').replace(')', '').replace(',', 'j')\
        .replace(' ', '')
    cfg['version'] = __version__
    cfg['git_rev'] = get_git_revision()

    return cfg

def generate_experiment_cfgs(exp_id):
    cfgs = []

    # -------------------------------------------------------------------------
    # SemiVL on VOC
    # -------------------------------------------------------------------------
    if exp_id == 40:
        n_repeat = 1
        splits = [92, 183, 366, 732, 1464]
        list_kwargs = [
            ### SemiVL
            dict(model='mmseg.vlm-vlg-aspp-s2p4-sk04-ftap-mcvitb', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss',
                 maskclip_consistency_lambda=[0.1, 0], mcc_conf_thresh=0.9, mcc_text='concept4_single', mcc_loss_reduce='mean_all'),
        ]
        for split, kwargs, _ in itertools.product(splits, list_kwargs, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                split=str(split),
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # Ablations on VOC
    # -------------------------------------------------------------------------
    elif exp_id == 41:
        n_repeat = 1
        splits = [92, 1464]
        list_kwargs = [
            # ### Original UniMatch (just for reference)
            # dict(model='dlv3p-r101', opt='original', lr=1e-3, backbone_lr_mult=10, criterion='CELoss',
            #      img_scale=None, crop_size=321, eval_mode='original'),
            ### UniMatch w/ ZegCLIP
            dict(model='mmseg.vlm-zegclip-rd-pt-vitb', lr=1e-4, backbone_lr_mult=10, criterion='mmseg'),
            ### UniMatch w/ ViT
            dict(model='mmseg.vlm-dlv3p-bn11-sk4-ft-tvit-in1k', lr=1e-4, backbone_lr_mult=0.001, criterion='CELoss'),
            ### + CLIP Init
            dict(model='mmseg.vlm-dlv3p-bn12-sk4-ft-mcvitb', lr=1e-4, backbone_lr_mult=0.001, criterion='CELoss'),
            ### + CLIP Init + SFT
            dict(model='mmseg.vlm-dlv3p-bn12-sk4-ftap-mcvitb', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
            ### + CLIP Init + SFT + VLDec
            dict(model='mmseg.vlm-vlg-aspp-s2p4-sk04-ftap-mcvitb', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss'),
            ### + CLIP Init + SFT + VLDec + CLIP Guid.
            dict(model='mmseg.vlm-vlg-aspp-s2p4-sk04-ftap-mcvitb', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss',
                 maskclip_consistency_lambda=[0.1, 0], mcc_conf_thresh=0.9, mcc_loss_reduce='mean_all'),
            ### + CLIP Init + SFT + VLDec + CLIP Guid. + ClsDef (already run in exp 40)
            # dict(model='mmseg.vlm-vlg-aspp-s2p4-sk04-ftap-mcvitb', lr=1e-4, backbone_lr_mult=0.01, criterion='CELoss',
            #      maskclip_consistency_lambda=[0.1, 0], mcc_conf_thresh=0.9, mcc_text='concept4_single', mcc_loss_reduce='mean_all'),
        ]
        for split, kwargs, _ in itertools.product(splits, list_kwargs, range(n_repeat)):
            cfg = config_from_vars(
                exp_id=exp_id,
                split=str(split),
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # SemiVL on COCO
    # -------------------------------------------------------------------------
    elif exp_id == 42:
        n_repeat = 1
        splits = ['1_512', '1_64', '1_128', '1_256', '1_32']
        list_kwargs = [
            # ### UniMatch w/ ViT
            # dict(model='mmseg.vlm-dlv3p-bn11-sk4-ft-tvit-in1k', lr=4e-4, backbone_lr_mult=0.001, criterion='CELoss'),
            ### SemiVL
            dict(model='mmseg.vlm-vlg-aspp-s2p4-sk04-ftap-mcvitb', lr=4e-4, backbone_lr_mult=0.001, criterion='CELoss',
                 maskclip_consistency_lambda=[0.1, 0], mcc_conf_thresh=0.9, mcc_loss_reduce='mean_all'),
        ]
        for split, kwargs, _ in itertools.product(splits, list_kwargs, range(n_repeat)):
            if 'vlg' in kwargs['model']:
                kwargs['n_nodes'], kwargs['n_gpus'], kwargs['batch_size'] = 1, 8, 1
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='coco',
                split=str(split),
                img_scale=None, 
                epochs=10,
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # SemiVL on ADE20K
    # -------------------------------------------------------------------------
    elif exp_id == 43:
        n_repeat = 1
        splits = ['1_128', '1_64', '1_32', '1_16', '1_8']
        kwargs_list = [
            # ### Original UniMatch
            # dict(model='dlv3p-r101', opt='original', lr=4e-3, eval_mode='original', img_scale=None, criterion='CELoss'),
            # ### UniMatch w/ ViT
            # dict(model='mmseg.vlm-dlv3p-bn11-sk4-ft-tvit-in1k', lr=4e-4, backbone_lr_mult=0.001, criterion='CELoss'),
            # ### SemiVL
            dict(model='mmseg.vlm-vlg-aspp-s2p4-sk04-ftap-mcvitb', lr=4e-4, backbone_lr_mult=0.001, criterion='CELoss',
                 maskclip_consistency_lambda=[0.1, 0], mcc_conf_thresh=0.9, mcc_loss_reduce='mean_all'),
        ]
        for kwargs, split, _ in itertools.product(kwargs_list, splits, range(n_repeat)):
            if 'vlg' in kwargs['model']:
                kwargs['n_nodes'], kwargs['n_gpus'], kwargs['batch_size'] = 1, 8, 1
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='ade',
                split=str(split),
                epochs=40,
                conf_thresh=0.95,
                criterion_u=kwargs['criterion'],
                **kwargs,
            )
            cfgs.append(cfg)
    # -------------------------------------------------------------------------
    # SemiVL on Cityscapes
    # -------------------------------------------------------------------------
    elif exp_id == 44:
        n_repeat = 1
        splits = ['1_30', '1_16', '1_8', '1_4', '1_2']
        kwargs_list = [
            # ### UniMatch w/ ViT
            # dict(model='mmseg.vlm-dlv3p-bn11-sk4-ft-tvit-in1k', lr=5e-5, backbone_lr_mult=0.1, criterion='CELoss'),
            # ### SemiVL
            dict(model='mmseg.vlm-vlg-aspp-s2p4-skr04-ftap-mcvitb', lr=5e-5, backbone_lr_mult=0.1, criterion='CELoss',
                 maskclip_consistency_lambda=[0.1, 0], mcc_conf_thresh=0.9, mcc_text='concept3_single', mcc_loss_reduce='mean_all',
                 text_embedding_variant='conceptavg3_single', renorm_clip_img=True, conv_enc_lr_mult=0.1),
        ]
        for kwargs, split, _ in itertools.product(kwargs_list, splits, range(n_repeat)):
            if 'vlg' in kwargs['model']:
                kwargs['n_nodes'], kwargs['n_gpus'], kwargs['batch_size'] = 1, 8, 1
            if 'criterion_u' not in kwargs:
                kwargs['criterion_u'] = kwargs['criterion']
            cfg = config_from_vars(
                exp_id=exp_id,
                dataset='cityscapes',
                split=str(split),
                img_scale=None,
                crop_size=801,
                epochs=None, iters=83760,  # ensure same #iters as in 1_16 with 80 epochs
                conf_mode='pixelavg',
                eval_every=10,
                eval_mode='sliding_window',
                **kwargs,
            )
            cfgs.append(cfg)
    else:
        raise NotImplementedError(f'Unknown id {exp_id}')

    return cfgs


def save_experiment_cfgs(exp_id):
    cfgs = generate_experiment_cfgs(exp_id)
    cfg_files = []
    for cfg in cfgs:
        cfg_file = f"configs/generated/exp-{cfg['exp']}/{cfg['name']}.yaml"
        os.makedirs(os.path.dirname(cfg_file), exist_ok=True)
        with open(cfg_file, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=None, sort_keys=False, indent=2)
        cfg_files.append(cfg_file)

    return cfgs, cfg_files

def run_command(command):
    p = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate experiment configs')
    parser.add_argument('--exp', type=int, help='Experiment id')
    parser.add_argument('--run', type=int, default=0, help='Run id')
    parser.add_argument('--ngpus', type=int, default=None, help='Override number of GPUs')
    args = parser.parse_args()

    cfgs, cfg_files = save_experiment_cfgs(args.exp)

    if args.ngpus is None:
        ngpus = cfgs[args.run]["n_gpus"]
    else:
        ngpus = args.ngpus

    cmd = f'bash scripts/train.sh {cfgs[args.run]["method"]} {cfg_files[args.run]} {ngpus}'
    print(cmd)
    run_command(cmd)

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

import argparse
import logging
import math
import os
import pprint
import shutil
import uuid
import time
from datetime import datetime

import mmcv
import torch
import torch.backends.cudnn as cudnn
import yaml
from matplotlib import pyplot as plt
from mmseg.core import build_optimizer
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.palettes import get_palette
from experiments import get_git_revision
from model.builder import build_model
from third_party.unimatch.supervised import evaluate
from third_party.unimatch.dataset.semi import SemiDataset
from datasets.classes import CLASSES
from third_party.unimatch.util.ohem import ProbOhemCrossEntropy2d
from third_party.unimatch.util.dist_helper import setup_distributed
from third_party.unimatch.util.utils import count_params, count_training_params, init_log
from utils.gen_code_archive import gen_code_archive
from utils.plot_utils import plot_data
from utils.train_utils import (DictAverageMeter, confidence_weighted_loss,
                               cutmix_img_, cutmix_mask)
from version import __version__


def compute_mc_loss(pred, mask, ign):
    l_mc = criterion_mc(pred, mask)
    if mcc_loss_reduce == 'mean_valid':
        l_mc = l_mc.sum() / (ign != 255).sum()
    if mcc_loss_reduce == 'mean_all':
        l_mc = l_mc.sum() / ign.numel()
    return l_mc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)

    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)
    labeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/labeled.txt'
    unlabeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/unlabeled.txt'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    mmcv.utils.get_logger('mmcv').setLevel('WARNING')

    rank, world_size = setup_distributed(port=args.port)
    if cfg['nccl_p2p_disable']:
        os.environ["NCCL_P2P_DISABLE"] = str(1)

    if rank == 0:
        timestr = datetime.now().strftime("%y%m%d-%H%M")
        uid = str(uuid.uuid4())[:5]
        run_name = f'{timestr}_{cfg["name"]}_v{__version__}_{uid}'.replace('.', '-')
        save_path = f'exp/exp-{cfg["exp"]}/{run_name}'
        os.makedirs(save_path, exist_ok=True)

        formatter = logging.Formatter(fmt='[%(asctime)s] [%(levelname)-8s] %(message)s')
        fileHandler = logging.FileHandler(f'{save_path}/debug.log')
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

        all_args = {**cfg, **vars(args), 
                    'labeled_id_path': labeled_id_path, 'unlabeled_id_path': unlabeled_id_path,
                    'ngpus': world_size, 'run_name': run_name, 'save_path': save_path,
                    'exec_git_rev': get_git_revision(), 'exec_version': __version__}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(save_path)
        
        shutil.copyfile(args.config, os.path.join(save_path, 'config.yaml'))
        with open(os.path.join(save_path, 'all_args.yaml'), 'w') as f:
            yaml.dump(all_args, f, default_flow_style=None, sort_keys=False, indent=2)
        gen_code_archive(save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    maskclip_consistency_lambda = cfg['maskclip_consistency_lambda']
    mcc_conf_thresh = cfg['mcc_conf_thresh']
    mcc_loss_reduce = cfg['mcc_loss_reduce']
    assert mcc_loss_reduce in ['mean', 'mean_valid', 'mean_all']
    assert cfg['use_fp']
    assert cfg['pleval']

    model = build_model(cfg)
    if 'optimizer' not in cfg:
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                        'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = build_optimizer(model, cfg['optimizer'])
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    
    if rank == 0:
        logger.info(model)
        logger.info(f'Total params: {count_params(model):.1f}M\n')
        if hasattr(model, 'backbone'):
            logger.info(f'Backbone params (training/total): {count_training_params(model.backbone):.1f}M/{count_params(model.backbone):.1f}M\n')
        if hasattr(model, 'decode_head'):
            logger.info(f'Decoder params (training/total): {count_training_params(model.decode_head):.1f}M/{count_params(model.decode_head):.1f}M\n')

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'mmseg':
        criterion_l = None
    else:
        raise ValueError(cfg['criterion_u']['name'])

    if cfg['criterion_u'] == 'CELoss':
        criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    elif cfg['criterion_u'] == 'mmseg':
        criterion_u = None
    else:
        raise ValueError(cfg['criterion_u'])

    if maskclip_consistency_lambda != 0:
        if mcc_loss_reduce == 'mean':
            criterion_mc = nn.CrossEntropyLoss(ignore_index=255).cuda(local_rank)
        elif mcc_loss_reduce in ['mean_valid', 'mean_all']:
            criterion_mc = nn.CrossEntropyLoss(ignore_index=255, reduction='none').cuda(local_rank)
        else:
            raise ValueError(mcc_loss_reduce)

    trainset_u = SemiDataset(cfg, 'train_u', id_path=unlabeled_id_path)
    trainset_l = SemiDataset(cfg, 'train_l', id_path=labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg, 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    palette = get_palette(cfg['dataset'])

    if cfg['iters'] is not None:
        assert cfg['epochs'] is None
        cfg['epochs'] = math.ceil(cfg['iters'] / len(trainloader_u))

    total_iters = len(trainloader_u) * cfg['epochs']
    scheduler_max_iters = cfg.get('scheduler_max_iters', total_iters)
    assert scheduler_max_iters >= total_iters
    if rank == 0:
        logger.info(f'Train for {cfg["epochs"]} epochs / {total_iters} iterations.')
    previous_best = 0.0
    epoch = -1
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        log_avg = DictAverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_w, img_s1, img_s2, ignore_mask, mix1, mix2),
                (img_w_other, img_s1_other, img_s2_other, ignore_mask_other, _, _)) in enumerate(loader):
            t0 = time.time()
            iters = epoch * len(trainloader_u) + i
            img_x = img_x.cuda()
            img_s1 = img_s1.cuda()
            img_s2 = img_s2.cuda()
            mask_x = mask_x.cuda()
            img_w = img_w.cuda()
            ignore_mask = ignore_mask.cuda()
            mix1 = mix1.cuda()
            mix2 = mix2.cuda()
            img_w_other = img_w_other.cuda()
            img_s1_other = img_s1_other.cuda()
            img_s2_other = img_s2_other.cuda()
            ignore_mask_other = ignore_mask_other.cuda()

            # CutMix images
            cutmix_img_(img_s1, img_s1_other, mix1)
            cutmix_img_(img_s2, img_s2_other, mix2)

            # Generate pseudo labels
            with torch.no_grad():
                model.eval()

                pred_w_other = model(img_w_other).detach()
                conf_w_other, mask_w_other = pred_w_other.softmax(dim=1).max(dim=1)

                if maskclip_consistency_lambda != 0:
                    mclip = model.module.forward_maskclip(
                        torch.cat((img_w, img_w_other)),
                        conf_tresh=mcc_conf_thresh)
                    mclip, mclip_other = mclip.split([img_w.shape[0], img_w_other.shape[0]])
                    mclip[ignore_mask == 255] = 255
                    mclip_other[ignore_mask_other == 255] = 255

            # Generate predictions
            model.train()

            preds, preds_fp = model(torch.cat((img_x, img_w)), need_fp=True)
            pred_x, pred_w = preds.chunk(2)
            _, pred_w_fp = preds_fp.chunk(2)

            pred_s1, pred_s2 = model(torch.cat((img_s1, img_s2))).chunk(2)

            pred_w = pred_w.detach()
            conf_w, mask_w = pred_w.softmax(dim=1).max(dim=1)

            # CutMix labels
            mask_w_mixed1 = cutmix_mask(mask_w, mask_w_other, mix1)
            mask_w_mixed2 = cutmix_mask(mask_w, mask_w_other, mix2)
            conf_w_mixed1 = cutmix_mask(conf_w, conf_w_other, mix1)
            conf_w_mixed2 = cutmix_mask(conf_w, conf_w_other, mix2)
            ignore_mask_mixed1 = cutmix_mask(ignore_mask, ignore_mask_other, mix1)
            ignore_mask_mixed2 = cutmix_mask(ignore_mask, ignore_mask_other, mix2)

            if maskclip_consistency_lambda != 0:
                mclip_mixed1 = cutmix_mask(mclip, mclip_other, mix1)
                mclip_mixed2 = cutmix_mask(mclip, mclip_other, mix2)

            # Supervised Loss
            if criterion_l is not None:
                loss_x = criterion_l(pred_x, mask_x)
            else:
                losses = model.module.decode_head.loss_decode({'pred_masks': pred_x}, mask_x)
                loss_x, log_vars_x = model.module._parse_losses(losses)

            # FixMatch 1 Loss
            if criterion_u is not None:
                loss_s1 = criterion_u(pred_s1, mask_w_mixed1)
                loss_s1 = confidence_weighted_loss(loss_s1, conf_w_mixed1, ignore_mask_mixed1, cfg)
            else:
                loss_s1, _ = model.module._parse_losses(
                    model.module.decode_head.loss_decode({'pred_masks': pred_s1}, mask_w_mixed1))
                conf_ratio = ((conf_w_mixed1 >= cfg['conf_thresh']) & (ignore_mask_mixed1 != 255)).sum().item() / \
                    (ignore_mask_mixed1 != 255).sum().item()
                loss_s1 *= conf_ratio
            if maskclip_consistency_lambda != 0:
                loss_mc_s1 = compute_mc_loss(pred_s1, mclip_mixed1, ignore_mask_mixed1)

            # FixMatch 2 Loss
            if criterion_u is not None:
                loss_s2 = criterion_u(pred_s2, mask_w_mixed2)
                loss_s2 = confidence_weighted_loss(loss_s2, conf_w_mixed2, ignore_mask_mixed2, cfg)
            else:
                loss_s2, _ = model.module._parse_losses(
                    model.module.decode_head.loss_decode({'pred_masks': pred_s2}, mask_w_mixed2))
                conf_ratio = ((conf_w_mixed2 >= cfg['conf_thresh']) & (ignore_mask_mixed2 != 255)).sum().item() / \
                    (ignore_mask_mixed2 != 255).sum().item()
                loss_s2 *= conf_ratio
            if maskclip_consistency_lambda != 0:
                loss_mc_s2 = compute_mc_loss(pred_s2, mclip_mixed2, ignore_mask_mixed2)

            # Feature Perturbation Loss
            if criterion_u is not None:
                loss_fp = criterion_u(pred_w_fp, mask_w)
                loss_fp = confidence_weighted_loss(loss_fp, conf_w, ignore_mask, cfg)
            else:
                loss_fp, _ = model.module._parse_losses(
                    model.module.decode_head.loss_decode({'pred_masks': pred_w_fp}, mask_w))
                conf_ratio = ((conf_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / \
                    (ignore_mask != 255).sum().item()
                loss_fp *= conf_ratio
            if maskclip_consistency_lambda != 0:
                loss_mc_fp = compute_mc_loss(pred_w_fp, mclip, ignore_mask)

            if maskclip_consistency_lambda != 0:
                if isinstance(maskclip_consistency_lambda, list) or isinstance(maskclip_consistency_lambda, tuple):
                    assert len(maskclip_consistency_lambda) == 2
                    prog = iters / total_iters
                    current_mcc_lambda = maskclip_consistency_lambda[0] * (1 - prog) + maskclip_consistency_lambda[1] * prog
                else:
                    current_mcc_lambda = maskclip_consistency_lambda
            loss = (loss_x + loss_s1 * 0.25 + loss_s2 * 0.25 + loss_fp * 0.5) / 2.0
            if maskclip_consistency_lambda != 0:
                loss = loss + loss_mc_s1 * 0.25 * current_mcc_lambda
                loss = loss + loss_mc_s2 * 0.25 * current_mcc_lambda
                loss = loss + loss_mc_fp * 0.5 * current_mcc_lambda

            torch.distributed.barrier()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if 'optimizer' not in cfg:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    lr = cfg['lr'] * (1 - k)
                else:
                    lr = cfg['lr'] * (1 - iters / scheduler_max_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            else:
                if iters < cfg['warmup_iters']:
                    k = (1 - iters / cfg['warmup_iters']) * (1 - cfg['warmup_ratio'])
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - k)
                else:
                    for group in optimizer.param_groups:
                        group['lr'] = group['initial_lr'] * (1 - iters / scheduler_max_iters) ** 0.9
                        # print(iters, group['initial_lr'], group['lr'], group['weight_decay'])

            # Logging
            log_avg.update({
                'train/iter_time': time.time() - t0,
                'train/loss_all': loss,
                'train/loss_x': loss_x,
                'train/loss_s1': loss_s1,
                'train/loss_s2': loss_s2,
                'train/loss_fp': loss_fp,
            })
            if maskclip_consistency_lambda != 0:
                log_avg.update({
                    'train/loss_mc_s1': loss_mc_s1,
                    'train/loss_mc_s2': loss_mc_s2,
                    'train/loss_mc_fp': loss_mc_fp,
                })

            if i % 100 == 0 and rank == 0:
                logger.info(f'Iters: {i} ' + str(log_avg))
                for k, v in log_avg.avgs.items():
                    writer.add_scalar(k, v, iters)

                log_avg.reset()

            if iters % len(trainloader_u) == 0 and rank == 0:
                print('Save debug images at iteration', iters)
                out_dir = os.path.join(save_path, 'debug')
                os.makedirs(out_dir, exist_ok=True)
                for b_i in range(img_x.shape[0]):
                    rows, cols = 3, 4
                    plot_dicts = [
                        dict(title='Image L', data=img_x[b_i], type='image'),
                        dict(title='Image S1', data=img_s1[b_i], type='image'),
                        dict(title='Image S2', data=img_s2[b_i], type='image'),
                        dict(title='Image FP', data=img_w[b_i], type='image'),
                        dict(title='Pred L', data=pred_x[b_i], type='prediction', palette=palette),
                        dict(title='Pred S1', data=pred_s1[b_i], type='prediction', palette=palette),
                        dict(title='Pred S2', data=pred_s2[b_i], type='prediction', palette=palette),
                        dict(title='Pred FP', data=pred_w_fp[b_i], type='prediction', palette=palette),
                        dict(title='GT L', data=mask_x[b_i], type='label', palette=palette),
                        dict(title='PL S1', data=mask_w_mixed1[b_i], type='label', palette=palette),
                        dict(title='PL S2', data=mask_w_mixed2[b_i], type='label', palette=palette),
                        dict(title='PL FP', data=mask_w[b_i], type='label', palette=palette),
                    ]
                    if maskclip_consistency_lambda != 0:
                        plot_dicts.extend([
                            None,
                            dict(title='MC S1', data=mclip_mixed1[b_i], type='label', palette=palette),
                            dict(title='MC S2', data=mclip_mixed2[b_i], type='label', palette=palette),
                            dict(title='MC FP', data=mclip[b_i], type='label', palette=palette),
                        ])
                        rows += 1
                    fig, axs = plt.subplots(
                        rows, cols, figsize=(2 * cols, 2 * rows), squeeze=False, 
                        gridspec_kw={'hspace': 0.1, 'wspace': 0, 'top': 0.95, 'bottom': 0, 'right': 1, 'left': 0})
                    for ax, plot_dict in zip(axs.flat, plot_dicts):
                        if plot_dict is not None:
                            plot_data(ax, **plot_dict)
                    plt.savefig(os.path.join(out_dir, f'{(iters):07d}_{rank}-{b_i}.png'))
                    plt.close()

        if epoch % cfg.get('eval_every_n_epochs', 1) == 0 or epoch == cfg['epochs'] - 1:
            eval_mode = cfg['eval_mode']
            mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

            if rank == 0:
                logger.info(run_name)
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }
                # torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))
                if is_best:
                    torch.save(checkpoint, os.path.join(save_path, 'best.pth'))

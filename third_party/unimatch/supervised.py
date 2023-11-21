import argparse
import logging
import os
import pprint
import shutil
import uuid
from version import __version__
from datetime import datetime
from utils.gen_code_archive import gen_code_archive

import torch
import numpy as np
from tqdm import tqdm
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import mmseg

from third_party.unimatch.dataset.semi import SemiDataset
from model.builder import build_model
from mmseg.core import build_optimizer
from experiments import get_git_revision
from datasets.classes import CLASSES
from third_party.unimatch.util.ohem import ProbOhemCrossEntropy2d
from third_party.unimatch.util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from third_party.unimatch.util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def predict(model, img, mask, mode, cfg, return_logits=False):
    if mode == 'padded_sliding_window':
        grid = cfg['crop_size']
        stride = cfg['stride']
        if stride < 1:
            stride = int(grid * stride)
        b, _, h, w = img.shape
        final = torch.zeros(b, cfg['nclass'], h, w).cuda()
        row = 0
        while row < h:
            col = 0
            while col < w:
                y1 = row
                y2 = min(h, row + grid)
                x1 = col
                x2 = min(w, col + grid)
                crop_h = y2 - y1
                crop_w = x2 - x1
                # print(y1, y2, x1, x2, crop_h, crop_w)
                cropped_img = torch.zeros((b, 3, grid, grid), device=img.device)
                cropped_img[:, :, :crop_h, :crop_w] = img[:, :, y1: y2, x1: x2]

                pred = model(cropped_img)
                final[:, :, y1: y2, x1: x2] += pred.softmax(dim=1)[:, :, :crop_h, :crop_w]
                col += stride
            row += stride

        pred = final.argmax(dim=1)

    elif mode == 'zegclip_sliding_window':
        h_stride, w_stride = cfg['stride'], cfg['stride']
        h_crop, w_crop = cfg['crop_size'], cfg['crop_size']
        batch_size, _, h_img, w_img = img.size()
        num_classes = cfg['nclass']
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = model(crop_img)
                preds += F.pad(crop_seg_logit,
                            (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        preds = preds / count_mat
        final = mmseg.ops.resize(
            preds,
            size=mask.shape[-2:],
            mode='bilinear',
            align_corners=True,
            warning=False)

        pred = final.argmax(dim=1)

    elif mode == 'sliding_window':
        grid = cfg['crop_size']
        b, _, h, w = img.shape
        final = torch.zeros(b, cfg['nclass'], h, w).cuda()
        row = 0
        while row < h:
            col = 0
            while col < w:
                pred = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                col += int(grid * 2 / 3)
            row += int(grid * 2 / 3)

        pred = final.argmax(dim=1)

    else:
        if mode == 'center_crop':
            h, w = img.shape[-2:]
            start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
            img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
            mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

        final = model(img)
        pred = final.argmax(dim=1)
        
    if return_logits:
        return pred, final
    else:
        return pred


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ['original', 'center_crop', 'padded_sliding_window', 'zegclip_sliding_window', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask, id in tqdm(loader, total=len(loader)):
            
            img = img.cuda()
            pred = predict(model, img, mask, mode, cfg)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU, iou_class


def main():
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.Loader)
    labeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/labeled.txt'
    unlabeled_id_path = f'splits/{cfg["dataset"]}/{cfg["split"]}/unlabeled.txt'

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

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

    model = build_model(cfg)
    if rank == 0:
        logger.info(model)
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    if 'optimizer' not in cfg:
        optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                        {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                        'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = build_optimizer(model, cfg['optimizer'])
        # print(len(optimizer.param_groups), 'param groups')
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
            # print(group['initial_lr'], group['lr'], group['weight_decay'])

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=('zegclip' in cfg['model']))

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'mmseg':
        criterion = None
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg, 'train_l', id_path=labeled_id_path)
    valset = SemiDataset(cfg, 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     previous_best = checkpoint['previous_best']
        
    #     if rank == 0:
    #         logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            if criterion is not None:
                loss = criterion(pred, mask)
            else:
                losses = model.module.decode_head.loss_decode({'pred_masks': pred}, mask)
                loss, log_vars = model.module._parse_losses(losses)
            
            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            if 'optimizer' not in cfg:
                lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            else:
                for group in optimizer.param_groups:
                    group['lr'] = group['initial_lr'] * (1 - iters / total_iters) ** 0.9
                    # print(iters, group['initial_lr'], group['lr'], group['weight_decay'])

            if i % 100 == 0 and rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))

        if (epoch+1) % cfg.get('eval_every_n_epochs', 1) == 0 or epoch+1 == cfg['epochs']:
            eval_mode = cfg['eval_mode']
            mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)

            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            # if rank == 0:
            #     checkpoint = {
            #         'model': model.state_dict(),
            #         'optimizer': optimizer.state_dict(),
            #         'epoch': epoch,
            #         'previous_best': previous_best,
            #     }
            #     torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))
            #     if is_best:
            #         torch.save(checkpoint, os.path.join(save_path, 'best.pth'))


if __name__ == '__main__':
    main()

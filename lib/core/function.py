# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..utils.transforms import flip_back
from .evaluation import accuracy, decode_preds, compute_nme
from ..utils.imutils import batch_with_heatmap

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    model.train()
    end = time.time()

    for i, (inp, target, meta) in enumerate(train_loader):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)

        loss = critertion(output, target)

        # optimzie
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.itme(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()


def validate(config, val_loader, model, criterion, output_dir,
             tb_log_dir, writer_dict, debug=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    acces = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()
    flip = config.TEST.FLIP_TEST
    nme_count = 0
    nme_batch_sum = 0
    gt_win, pred_win = None, None
    with torch.no_grad():
        end = time.time()
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()

            if flip:
                # flip W
                flip_input = torch.flip(inp, dim=[3])
                flip_output = model(flip_input)
                # [-1] ??
                flip_output = flip_back(flip_output[-1].data.cpu())
                score_map += flip_output
            # loss
            loss = criterion(output, target)

            # accuracy
            acc = accuracy(score_map, target.cpu(), i)
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            nme_batch_sum = nme_batch_sum + compute_nme(preds, meta['pts'])
            nme_count = nme_count + preds.size(0)

            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            if debug:  # and epoch % args.display == 0
                gt_batch_img = batch_with_heatmap(inp, target)
                pred_batch_img = batch_with_heatmap(inp, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    plt.title('Val-Groundtruth')
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    plt.title('Prediction')
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            losses.update(loss.item(), inp.size(0))
            acces.update(acc.item(), inp.size(0))

            batch_time.update(time.time()-end)
            end = time.time()

    return losses.avg, acces.avg, predictions, nme_batch_sum / nme_count


def evaluate(config, val_loader, model, criterion, debug=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()
    acces = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    gt_win, pred_win = None, None
    end = time.time()
    flip = config.TEST.FLIP_TEST

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()

            if flip:
                # flip W
                flip_input = torch.flip(inp, dim=[3])
                flip_output = model(flip_input)
                # [-1] ??
                flip_output = flip_back(flip_output[-1].data.cpu())
                score_map += flip_output
            # loss
            loss = criterion(output, target)

            # accuracy
            acc = accuracy(score_map, target.cpu(), i)
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

            # NME
            nme_temp = compute_nme(preds, meta['pts'])

            if nme_temp > 0.08:
                count_failure_008 += 1

            if nme_temp > 0.10:
                count_failure_010 += 1

            nme_batch_sum += nme_temp
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            if debug:  # and epoch % args.display == 0
                gt_batch_img = batch_with_heatmap(inp, target)
                pred_batch_img = batch_with_heatmap(inp, score_map)
                if not gt_win or not pred_win:
                    plt.subplot(121)
                    plt.title('Val-Groundtruth')
                    gt_win = plt.imshow(gt_batch_img)
                    plt.subplot(122)
                    plt.title('Prediction')
                    pred_win = plt.imshow(pred_batch_img)
                else:
                    gt_win.set_data(gt_batch_img)
                    pred_win.set_data(pred_batch_img)
                plt.pause(.05)
                plt.draw()

            # measure accuracy and record loss
            losses.update(loss.item(), inp.size(0))
            acces.update(acc[0], inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print("Evaluation Finished...")
    print('count_failure_008 = ', count_failure_008)
    print('count_failure_010 = ', count_failure_010)
    print('nme_count = ', nme_count)

    return losses.avg, predictions, nme_batch_sum / nme_count, \
        count_failure_008 / nme_count, count_failure_010 / nme_count




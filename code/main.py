# -*- coding: utf-8 -*-

import argparse
import os
import shutil
import time
import numpy as np
import random
from collections import OrderedDict
import torch
import torch.backends.cudnn as cudnn
from callbacks import AverageMeter
from data_utils.causal_data_loader_frames import VideoFolder
from utils import save_results
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Counterfactual CAR')

# Path, dataset and log related arguments
parser.add_argument('--root_frames', type=str, default='/mnt/data1/home/sunpengzhan/sth-sth-v2/',
                    help='path to the folder with frames')
parser.add_argument('--json_data_train', type=str, default='../data/dataset_splits/compositional/train.json',
                    help='path to the json file with train video meta data')
parser.add_argument('--json_data_val', type=str, default='../data/dataset_splits/compositional/validation.json',
                    help='path to the json file with validation video meta data')
parser.add_argument('--json_file_labels', type=str, default='../data/dataset_splits/compositional/labels.json',
                    help='path to the json file with ground truth labels')

parser.add_argument('--dataset', default='smth_smth',
                    help='which dataset to train')
parser.add_argument('--logname', default='my_method',
                    help='name of the experiment for checkpoints and logs')
parser.add_argument('--print_freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 20)')
parser.add_argument('--ckpt', default='./ckpt',
                    help='folder to output checkpoints')
parser.add_argument('--resume_vision', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_coord', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume_fusion', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

# model, image&feature dim and training related arguments
parser.add_argument('--model_vision', default='rgb_roi')
parser.add_argument('--model_coord', default='interaction')
parser.add_argument('--model_fusion', default='concat_fusion')
parser.add_argument('--fusion_function', default='fused_sum', type=str,
                    help='function for fusing activations from each branch')

parser.add_argument('--img_feature_dim', default=512, type=int, metavar='N',
                    help='intermediate feature dimension for image-based features')
parser.add_argument('--coord_feature_dim', default=512, type=int, metavar='N',
                    help='intermediate feature dimension for coord-based features')
parser.add_argument('--size', default=224, type=int, metavar='N',
                    help='primary image input size')
parser.add_argument('--num_boxes', default=4, type=int,
                    help='num of boxes for each image')
parser.add_argument('--num_frames', default=16, type=int,
                    help='num of frames for the model')

parser.add_argument('--num_classes', default=174, type=int,
                    help='num of class in the model')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=None, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', '-b', default=16, type=int,
                    metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr_steps', default=[24, 35, 45], type=float, nargs="+",
                    metavar='LRSteps', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight_decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--clip_gradient', '-cg', default=5, type=float,
                    metavar='W', help='gradient norm clipping (default: 5)')
parser.add_argument('--search_stride', type=int, default=5, help='test performance every n strides')

# train mode, hardware setting and others related arguments
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--cf_inference_group', action='store_true', help='counterfactual inference model on validation set')
parser.add_argument('--parallel', default=True, type=bool,
                    help='whether or not train with multi GPUs')
parser.add_argument('--gpu_index', type=str, default='0, 1, 2, 3', help='the index of gpu you want to use')


best_loss = 1000000


def main():

    global args, best_loss
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    print(args)

    # create vision model
    if args.model_vision == 'global_i3d':
        from model.model_lib import VideoGlobalModel as RGBModel
        print('global_i3d loaded!!')
    elif args.model_vision == 'rgb_roi':
        from model.model_lib import BboxVisualModel as RGBModel
        print('rgb_roi loaded!!')
    else:
        print("no such a vision model!")

    # create coord model
    if args.model_coord == 'interaction':
        from model.model_lib import BboxInteractionLatentModel as BboxModel
        print('interaction loaded!!')
    else:
        print("no such a coordinate model!")

    # create fusion model
    if args.model_fusion == 'concat_fusion':
        from model.model_lib import ConcatFusionModel as FusionModel
        print('concat_fusion loaded!!')
    else:
        print('no such a fusion model!')

    # load model branch
    vision_model = RGBModel(args)
    coord_model = BboxModel(args)
    fusion_model = FusionModel(args)

    # create the fusion function for the activation of three branches
    if args.fusion_function == 'fused_sum':
        from fusion_function import logsigsum as fusion_func
        print('fused_sum loaded!!')
    elif args.fusion_function == 'naive_sum':
        from fusion_function import naivesum as fusion_func
        print('naive_sum loaded!!')
    else:
        print('no such a fusion function!')

    fusion_function = fusion_func()

    if args.parallel:
        vision_model = torch.nn.DataParallel(vision_model).cuda()
        coord_model = torch.nn.DataParallel(coord_model).cuda()
        fusion_model = torch.nn.DataParallel(fusion_model).cuda()
    else:
        vision_model = vision_model.cuda()
        coord_model = coord_model.cuda()
        fusion_model = fusion_model.cuda()

    # optionally resume vision model from a checkpoint
    if args.resume_vision:
        assert os.path.isfile(args.resume_vision), "No checkpoint found at '{}'".format(args.resume_vision)
        print("=> loading checkpoint '{}'".format(args.resume_vision))
        checkpoint = torch.load(args.resume_vision)
        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        vision_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume_vision, checkpoint['epoch']))
    # optionally resume coord model from a checkpoint
    if args.resume_coord:
        assert os.path.isfile(args.resume_coord), "No checkpoint found at '{}'".format(args.resume_coord)
        print("=> loading checkpoint '{}'".format(args.resume_coord))
        checkpoint = torch.load(args.resume_coord)
        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        coord_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume_coord, checkpoint['epoch']))
    if args.resume_fusion:
        assert os.path.isfile(args.resume_fusion), "No checkpoint found at '{}'".format(args.resume_fusion)
        print("=> loading checkpoint '{}'".format(args.resume_fusion))
        checkpoint = torch.load(args.resume_fusion)
        if args.start_epoch is None:
            args.start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        fusion_model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.resume_fusion, checkpoint['epoch']))

    if args.start_epoch is None:
        args.start_epoch = 0

    cudnn.benchmark = True

    # create training and validation dataset
    dataset_train = VideoFolder(root=args.root_frames,
                                num_boxes=args.num_boxes,
                                file_input=args.json_data_train,
                                file_labels=args.json_file_labels,
                                frames_duration=args.num_frames,
                                args=args,
                                is_val=False,
                                if_augment=True,
                                )
    dataset_val = VideoFolder(root=args.root_frames,
                              num_boxes=args.num_boxes,
                              file_input=args.json_data_val,
                              file_labels=args.json_file_labels,
                              frames_duration=args.num_frames,
                              args=args,
                              is_val=True,
                              if_augment=True,
                              )

    # create training and validation loader
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, drop_last=True,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        dataset_val, drop_last=True,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False
    )

    model_list = [vision_model, coord_model, fusion_model]

    optimizer_vision = torch.optim.SGD(filter(lambda p: p.requires_grad, vision_model.parameters()),
                                       momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_coord = torch.optim.SGD(filter(lambda p: p.requires_grad, coord_model.parameters()),
                                      momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_fusion = torch.optim.SGD(filter(lambda p: p.requires_grad, fusion_model.parameters()),
                                      momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_list = [optimizer_vision, optimizer_coord, optimizer_fusion]
    criterion = torch.nn.CrossEntropyLoss()
    search_list = np.linspace(0.0, 1.0, 11)

    # factual inference (vanilla test stage)
    if args.evaluate:
        validate(val_loader, model_list, fusion_function, criterion, class_to_idx=dataset_val.classes_dict)
        return

    # Counterfactual inference by trying a list of hyperparameter
    if args.cf_inference_group:
        cf_inference_group(val_loader, model_list, fusion_function, search_list,
                           class_to_idx=dataset_val.classes_dict)
        return

    print('training begin...')
    for epoch in tqdm(range(args.start_epoch, args.epochs)):

        adjust_learning_rate(optimizer_vision, epoch, args.lr_steps, 'vision')
        adjust_learning_rate(optimizer_coord, epoch, args.lr_steps, 'coord')
        adjust_learning_rate(optimizer_fusion, epoch, args.lr_steps, 'fusion')
        # train for one epoch
        train(train_loader, model_list, fusion_function, optimizer_list, epoch, criterion)

        if (epoch+1) >= 30 and (epoch + 1) % args.search_stride == 0:
            loss = validate(val_loader, model_list, fusion_function, criterion,
                            epoch=epoch, class_to_idx=dataset_val.classes_dict)
        else:
            loss = 100
        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': vision_model.state_dict(),
                'best_loss': best_loss,
            },
            is_best,
            os.path.join(args.ckpt,  '{}_{}'.format(args.model_vision, args.logname)))
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': coord_model.state_dict(),
                'best_loss': best_loss,
            },
            is_best,
            os.path.join(args.ckpt, '{}_{}'.format(args.model_coord, args.logname)))
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': fusion_model.state_dict(),
                'best_loss': best_loss,
            },
            is_best,
            os.path.join(args.ckpt, '{}_{}'.format(args.model_fusion, args.logname)))


def train(train_loader, model_list, fusion_function,
          optimizer_list, epoch, criterion):

    global args
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()

    # load three model branches
    [vision_model, coord_model, fusion_model] = model_list

    # load four optimizers, including the one designed for uniform assumption
    [optimizer_vision, optimizer_coord, optimizer_fusion] = optimizer_list

    # switch to train mode
    vision_model.train()
    coord_model.train()
    fusion_model.train()

    end = time.time()
    for i, (global_img_tensors, box_tensors, box_categories, video_label) in enumerate(train_loader):

        data_time.update(time.time() - end)
        
        # obtain the activation and vision features from vision branch
        output_vision, feature_vision = vision_model(global_img_tensors.cuda(), box_categories, box_tensors.cuda(), video_label)
        output_vision = output_vision.view((-1, len(train_loader.dataset.classes)))

        # obtain the activation and coordinate features from coordinate branch
        output_coord, feature_coord = coord_model(global_img_tensors, box_categories.cuda(), box_tensors.cuda(), video_label)
        output_coord = output_coord.view((-1, len(train_loader.dataset.classes)))

        # detach the computation graph, avoid the gradient confusion
        feature_vision_detached = feature_vision.detach()
        feature_coord_detached = feature_coord.detach()

        # obtain the activation of fusion branch
        output_fusion = fusion_model(feature_vision_detached.cuda(), feature_coord_detached.cuda())
        output_fusion = output_fusion.view((-1, len(train_loader.dataset.classes)))
        output_factual = fusion_function(output_vision, output_coord, output_fusion)

        # loss_fusion is the loss of output_fusion(fused, obtained from the fusion_function)
        loss_vision = criterion(output_vision, video_label.long().cuda())
        loss_coord = criterion(output_coord, video_label.long().cuda())
        loss_fusion = criterion(output_fusion, video_label.long().cuda())
        loss_factual = criterion(output_factual, video_label.long().cuda())

        # Measure the accuracy of the sum of three branch activation results
        acc1, acc5 = accuracy(output_factual.cpu(), video_label, topk=(1, 5))

        # record the accuracy and loss
        losses.update(loss_factual.item(), global_img_tensors.size(0))

        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))

        # refresh the optimizer
        optimizer_vision.zero_grad()
        optimizer_coord.zero_grad()
        optimizer_fusion.zero_grad()

        loss = loss_vision + loss_coord + loss_factual
        loss.backward()
        if args.clip_gradient is not None:
            torch.nn.utils.clip_grad_norm_(vision_model.parameters(), args.clip_gradient)

        # update the parameter
        optimizer_vision.step()
        optimizer_coord.step()
        optimizer_fusion.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   acc_top1=acc_top1, acc_top5=acc_top5))


def validate(val_loader, model_list, fusion_function, criterion,
             epoch=None, class_to_idx=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    logits_matrix = []
    targets_list = []

    # unpack three models
    [vision_model, coord_model, fusion_model] = model_list

    # switch to evaluate mode
    vision_model.eval()
    coord_model.eval()
    fusion_model.eval()

    end = time.time()
    for i, (global_img_tensors, box_tensors, box_categories, video_label) in enumerate(val_loader):
        # compute output
        with torch.no_grad():

            output_vision, feature_vision = vision_model(global_img_tensors.cuda(), box_categories, box_tensors.cuda(), video_label)
            output_vision = output_vision.view((-1, len(val_loader.dataset.classes)))

            output_coord, feature_coord = coord_model(global_img_tensors, box_categories.cuda(), box_tensors.cuda(), video_label)
            output_coord = output_coord.view((-1, len(val_loader.dataset.classes)))

            # detach the computation graph, avoid the gradient confusion
            feature_vision_detached = feature_vision.detach()
            feature_coord_detached = feature_coord.detach()

            # obtain the activation of fusion branch
            output_fusion = fusion_model(feature_vision_detached.cuda(), feature_coord_detached.cuda())
            output_fusion = output_fusion.view((-1, len(val_loader.dataset.classes)))

            # fuse three outputs
            output_factual = fusion_function(output_vision, output_coord, output_fusion)

            # warning: loss_fusion is the loss of output_fusion(fused, obtained from the fusion_function)
            loss_vision = criterion(output_vision, video_label.long().cuda())
            loss_coord = criterion(output_coord, video_label.long().cuda())
            loss_fusion = criterion(output_factual, video_label.long().cuda())

            # statistic result from fusion_branch or value after fusion function
            output = output_factual
            loss = loss_vision

            acc1, acc5 = accuracy(output.cpu(), video_label, topk=(1, 5))

            if args.evaluate:
                logits_matrix.append(output.cpu().data.numpy())
                targets_list.append(video_label.cpu().numpy())

        # measure accuracy and record loss
        losses.update(loss.item(), global_img_tensors.size(0))
        acc_top1.update(acc1.item(), global_img_tensors.size(0))
        acc_top5.update(acc5.item(), global_img_tensors.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc1 {acc_top1.val:.1f} ({acc_top1.avg:.1f})\t'
                  'Acc5 {acc_top5.val:.1f} ({acc_top5.avg:.1f})\t'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                acc_top1=acc_top1, acc_top5=acc_top5,
            ))

    if args.evaluate:
        logits_matrix = np.concatenate(logits_matrix)
        targets_list = np.concatenate(targets_list)
        save_results(logits_matrix, targets_list, class_to_idx, args)

    return losses.avg


def cf_inference_group(val_loader, model_list, fusion_function, search_list, class_to_idx=None):
    batch_time = AverageMeter()
    search_length = len(search_list)
    search_dict = {}
    for i in range(search_length):
        search_dict['acc_1_alpha_{}'.format(round(search_list[i], 1))] = AverageMeter()
        search_dict['acc_5_alpha_{}'.format(round(search_list[i], 1))] = AverageMeter()

    [vision_model, coord_model, fusion_model] = model_list

    # switch to evaluate mode
    vision_model.eval()
    coord_model.eval()
    fusion_model.eval()

    end = time.time()
    for i, (global_img_tensors, box_tensors, box_categories, video_label) in enumerate(val_loader):
        # compute output
        with torch.no_grad():

            # factual inference
            output_vision, feature_vision = vision_model(global_img_tensors.cuda(), box_categories, box_tensors.cuda(),
                                                         video_label)
            output_vision = output_vision.view((-1, len(val_loader.dataset.classes)))

            output_coord, feature_coord = coord_model(global_img_tensors, box_categories.cuda(), box_tensors.cuda(),
                                                         video_label)
            output_coord = output_coord.view((-1, len(val_loader.dataset.classes)))

            # obtain the activation of fusion branch
            output_fusion = fusion_model(feature_vision.cuda(), feature_coord.cuda())
            output_fusion = output_fusion.view((-1, len(val_loader.dataset.classes)))

            # fuse three outputs
            output_factual = fusion_function(output_vision, output_coord, output_fusion)

            # counterfactual inference
            output_vision_subtrahend = output_vision
            output_counterfactual = fusion_function(output_vision_subtrahend, torch.tensor(0.0), torch.tensor(0.0))

            for j in range(search_length):

                weight = search_list[j]
                output_debiased = output_factual - output_counterfactual * weight
                acc1, acc5 = accuracy(output_debiased.cpu(), video_label, topk=(1, 5))

                search_dict['acc_1_alpha_{}'.format(round(search_list[j], 1))].update(acc1.item(), global_img_tensors.size(0))
                search_dict['acc_5_alpha_{}'.format(round(search_list[j], 1))].update(acc5.item(), global_img_tensors.size(0))
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0 or i + 1 == len(val_loader):
            print('Cf-Inference: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Acc1_0.0 {acc_top1_00.val:.1f} ({acc_top1_00.avg:.1f})\t'
                  'Acc1_0.2 {acc_top1_02.val:.1f} ({acc_top1_02.avg:.1f})\t'
                  'Acc1_0.5 {acc_top1_05.val:.1f} ({acc_top1_05.avg:.1f})\t'
                  'Acc1_0.8 {acc_top1_08.val:.1f} ({acc_top1_08.avg:.1f})\t'
                  'Acc1_1.0 {acc_top1_10.val:.1f} ({acc_top1_10.avg:.1f})'.format(
                i, len(val_loader), batch_time=batch_time, acc_top1_00=search_dict['acc_1_alpha_0.0'],
                acc_top1_02=search_dict['acc_1_alpha_0.2'], acc_top1_05=search_dict['acc_1_alpha_0.5'],
                acc_top1_08=search_dict['acc_1_alpha_0.8'], acc_top1_10=search_dict['acc_1_alpha_1.0']))

    for k in range(search_length):
        print(search_list[k], search_dict['acc_1_alpha_{}'.format(round(search_list[k], 1))].avg,
              search_dict['acc_5_alpha_{}'.format(round(search_list[k], 1))].avg)

    return


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')


def adjust_learning_rate(optimizer, epoch, lr_steps, branch_name=None):
    """Sets the learning rate to the initial LR decayed by 10"""

    decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
    lr = args.lr * decay
    if branch_name == 'vision':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.8
    elif branch_name == 'coord':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif branch_name == 'fusion':
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()

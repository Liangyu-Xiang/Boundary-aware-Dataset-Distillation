# original code: https://github.com/dyhan0920/PyramidNet-PyTorch/blob/master/train.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
from data import load_data, MEANS, STDS
from misc.utils import random_indices, rand_bbox, AverageMeter, accuracy, get_time, Plotter
from efficientnet_pytorch import EfficientNet
import time
import warnings
import swanlab
from resnet import resnet18



warnings.filterwarnings("ignore")
model_names = sorted(
    name for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name]))

mean_torch = {}
std_torch = {}
for key, val in MEANS.items():
    mean_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)
for key, val in STDS.items():
    std_torch[key] = torch.tensor(val, device='cuda').reshape(1, len(val), 1, 1)




def define_model(args, nclass, logger=None, size=None):
    """Define neural network models
    """
    if size == None:
        size = args.size

    if args.net_type == 'resnet':
        model = RN.ResNet(args.dataset,
                          args.depth,
                          nclass,
                          norm_type=args.norm_type,
                          size=size,
                          nch=args.nch)
    elif args.net_type == 'resnet_ap':
        model = RNAP.ResNetAP(args.dataset,
                              args.depth,
                              nclass,
                              width=args.width,
                              norm_type=args.norm_type,
                              size=size,
                              nch=args.nch)
    elif args.net_type == 'efficient':
        model = EfficientNet.from_name('efficientnet-b0', num_classes=nclass)
    elif args.net_type == 'densenet':
        model = DN.densenet_cifar(nclass)
    elif args.net_type == 'convnet':
        width = int(128 * args.width)
        model = CN.ConvNet(nclass,
                           net_norm=args.norm_type,
                           net_depth=args.depth,
                           net_width=width,
                           channel=args.nch,
                           im_size=(args.size, args.size))
    else:
        raise Exception('unknown network architecture: {}'.format(args.net_type))

    if logger is not None:
        logger(f"=> creating model {args.net_type}-{args.depth}, norm: {args.norm_type}")

    return model


def main(args, logger, repeat=1):
    if args.seed >= 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    cudnn.benchmark = True
    logger(f"ImageNet directory: {args.imagenet_dir[0]}")
    _, train_loader, val_loader, nclass = load_data(args)

    best_acc_l = []
    acc_l = []
    for i in range(repeat):
        logger(f"Repeat: {i+1}/{repeat}")
        plotter = Plotter(args.save_dir, args.epochs, idx=i)
        model = define_model(args, nclass, logger)

        model_teacher = None

        best_acc, acc = train(args, model, train_loader, val_loader, model_teacher, plotter, logger)
        best_acc_l.append(best_acc)
        acc_l.append(acc)

    # swanlab.log({f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_l):.1f} {np.std(best_acc_l):.1f}'})
    logger(f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_l):.1f} {np.std(best_acc_l):.1f}')


def train(args, model, train_loader, val_loader, model_teacher=None, plotter=None, logger=None):
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0

    pretrained = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/imagenet10/resnet18in_resnet18imagewoof_cut/checkpoint.pth.tar"
    cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)
    # TODO: optimizer scheduler steps

    model = model.cuda()
    if model_teacher is not None:
        model_teacher = model_teacher.cuda()
    logger(f"Start training with base augmentation and {args.mixup} mixup")

    # Start training and validation
    for epoch in range(1):
        acc1_tr, _, loss_tr, train_confusion = train_epoch(args,
                                                           train_loader,
                                                           model,
                                                           criterion,
                                                           optimizer,
                                                           model_teacher,
                                                           epoch,
                                                           logger,
                                                           mixup=args.mixup)
        if logger is not None:
            logger(f'(Train) Confusion matrix epoch {epoch}:\n{train_confusion}')

        confusion_save_dir = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/test"
        os.makedirs(confusion_save_dir, exist_ok=True)
        confusion_path = os.path.join(confusion_save_dir, f"confusion_matrix_epoch_{epoch}.npy")
        np.save(confusion_path, train_confusion.cpu().numpy())
        if logger is not None:
            logger(f'(Train) Saved confusion matrix to {confusion_path}')

        # if epoch % args.epoch_print_freq == 0:
        #     acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch, logger)

        #     if plotter != None:
        #         plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

        #     is_best = acc1 > best_acc1
        #     if is_best:
        #         best_acc1 = acc1
        #         best_acc5 = acc5
        #         if logger != None and args.verbose == True:
        #             logger(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')
                # swanlab.log({"best_acc5": best_acc5, "best_acc1": best_acc1}, step=epoch)

        # if args.save_ckpt and (is_best or (epoch == args.epochs)):
        #     state = {
        #         'epoch': epoch,
        #         'arch': args.net_type,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'best_acc5': best_acc5,
        #         'optimizer': optimizer.state_dict(),
        #     }
        #     save_checkpoint(args.save_dir, state, is_best)
        # scheduler.step()

    return best_acc1, acc1

def distillation_loss(logits_s, logits_t, temperature):
    log_pred_student = F.log_softmax(logits_s / temperature, dim=1)
    pred_teacher = F.softmax(logits_t / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def evidential_criterion(logits, target):
    evidence = torch.exp(logits)
    alpha = evidence + 1
    labels_1hot = torch.zeros_like(logits).scatter_(-1, target.unsqueeze(-1), 1)
    S = torch.sum(alpha, dim=-1, keepdim=True)
    loss_ce = torch.sum(labels_1hot * (torch.digamma(S)-torch.digamma(alpha)), dim=-1).mean()
    return loss_ce

def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                model_teacher=None,
                epoch=0,
                logger=None,
                mixup='vanilla',
                n_data=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    num_classes = args.nclass
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)

    model.train()

    end = time.time()
    num_exp = 0
    for i, (input, target) in enumerate(train_loader):
        if train_loader.device == 'cpu':
            input = input.cuda()
            target = target.cuda()

        data_time.update(time.time() - end)

        r = np.random.rand(1)
        if r < args.mix_p and mixup == 'cut':
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = random_indices(target, nclass=args.nclass)

            target_b = target[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input[rand_index, :, bbx1:bbx2, bby1:bby2]
            ratio = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))

            output = model(input)
            loss = criterion(output, target) * ratio + criterion(output, target_b) * (1. - ratio)
            if model_teacher is not None:
                with torch.no_grad():
                    teacher_output = model_teacher(input)[0]
                loss += distillation_loss(output, teacher_output)
            # loss = evidential_criterion(output, target)* ratio + evidential_criterion(output, target_b) * (1. - ratio)
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)

        with torch.no_grad():
            pred = output.argmax(dim=1)
            target_cpu = target.detach().view(-1).long().cpu()
            pred_cpu = pred.view(-1).long().cpu()
            indices = target_cpu * num_classes + pred_cpu
            counts = torch.bincount(indices, minlength=num_classes * num_classes)
            confusion += counts.view(num_classes, num_classes) # confusion[i, j] 表示：真实类别 i 被预测为类别 j 的样本数

        


        # measure accuracy and record loss
        # acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        # losses.update(loss.item(), input.size(0))
        # top1.update(acc1.item(), input.size(0))
        # top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        num_exp += len(target)
        if (n_data > 0) and (num_exp >= n_data):
            break

    if (epoch % args.epoch_print_freq == 0) and (logger is not None) and args.verbose == True:
        logger(
            '(Train) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    # swanlab.log({"train_acc1": top1.avg, "train_acc5": top5.avg, "train_loss": losses.avg}, step=epoch)
    return top1.avg, top5.avg, losses.avg, confusion


def validate(args, val_loader, model, criterion, epoch, logger=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))

        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    if logger is not None and args.verbose == True:
        logger(
            '(Test ) [Epoch {0}/{1}] {2} Top1 {top1.avg:.1f}  Top5 {top5.avg:.1f}  Loss {loss.avg:.3f}'
            .format(epoch, args.epochs, get_time(), top1=top1, top5=top5, loss=losses))
    # swanlab.log({"test_acc1": top1.avg, "test_acc5": top5.avg, "test_loss": losses.avg}, step=epoch)
    return top1.avg, top5.avg, losses.avg


def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        # checkpoint['state_dict'] = dict(
        #     (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        # cur_epoch = checkpoint['epoch']
        cur_epoch = 0
        best_acc1 = checkpoint['best_acc1']
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}'(epoch: {}, best acc1: {}%)".format(
            path, cur_epoch, checkpoint['best_acc1']))
    else:
        print("=> no checkpoint found at '{}'".format(path))
        cur_epoch = 0
        best_acc1 = 100

    return cur_epoch, best_acc1


def save_checkpoint(save_dir, state, is_best):
    os.makedirs(save_dir, exist_ok=True)
    if is_best:
        ckpt_path = os.path.join(save_dir, 'model_best.pth.tar')
    else:
        ckpt_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, ckpt_path)
    print("checkpoint saved! ", ckpt_path)


if __name__ == '__main__':
    from misc.utils import Logger
    from argument import args

    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")
    if not args.test:
        run = swanlab.init(
            project= "DDresnetap-10",
            experiment_name= args.tag,
            x_axis="epoch"
        )
    main(args, logger, args.repeat)

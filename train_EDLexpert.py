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


class EvidentialLoss(nn.Module):
    def __init__(self, num_classes, lambda_kl=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl

    def forward(self, outputs, targets):
        """
        outputs: 网络原始输出 (B, K)，不经过 softmax
        targets: 标签 (B,) 或 (B, K) one-hot
        """

        if targets.dim() == 1:
            targets = F.one_hot(targets, self.num_classes).float()

        # ✅ 1. 证据必须非负
        evidence = F.softplus(outputs)   # e >= 0
        alpha = evidence + 1.0           # Dirichlet 参数

        S = torch.sum(alpha, dim=1, keepdim=True)

        # ✅ 2. 负对数似然项
        loglikelihood = torch.sum(
            targets * (torch.log(S) - torch.log(alpha)),
            dim=1
        )

        # ✅ 3. KL 正则项：Dir(alpha) || Dir(1)
        kl = self._kl_divergence(alpha)

        # loss = loglikelihood + self.lambda_kl * kl
        loss = loglikelihood
        return loss.mean()

    def _kl_divergence(self, alpha):
        K = alpha.size(1)
        S = torch.sum(alpha, dim=1, keepdim=True)

        log_B_alpha = torch.lgamma(S) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        log_B_uni = torch.lgamma(torch.tensor(K, device=alpha.device).float())

        digamma_sum = torch.digamma(S)
        digamma_alpha = torch.digamma(alpha)

        kl = torch.sum(
            (alpha - 1) * (digamma_alpha - digamma_sum),
            dim=1, keepdim=True
        )

        return kl.squeeze() + log_B_alpha.squeeze() + log_B_uni




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

        best_acc, acc = train(args, model, train_loader, val_loader, plotter, logger)
        best_acc_l.append(best_acc)
        acc_l.append(acc)

    swanlab.log({f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_l):.1f} {np.std(best_acc_l):.1f}'})
    logger(f'\n(Repeat {repeat}) Best, last acc: {np.mean(best_acc_l):.1f} {np.std(best_acc_l):.1f}')


def train(args, model, train_loader, val_loader, plotter=None, logger=None):
    # criterion = nn.CrossEntropyLoss().cuda()
    criterion = EvidentialLoss(num_classes=args.nclass, lambda_kl=1.0).cuda()
    optimizer = optim.SGD(model.parameters(),
                          args.lr,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2 * args.epochs // 3, 5 * args.epochs // 6], gamma=0.2)

    # Load pretrained
    cur_epoch, best_acc1, best_acc5, acc1, acc5 = 0, 0, 0, 0, 0

    if args.pretrained:
        pretrained = "{}/{}".format(args.save_dir, 'checkpoint.pth.tar')
        cur_epoch, best_acc1 = load_checkpoint(pretrained, model, optimizer)
        # TODO: optimizer scheduler steps

    model = model.cuda()

    logger(f"Start training with base augmentation and {args.mixup} mixup")

    # Start training and validation
    for epoch in range(cur_epoch + 1, args.epochs + 1):
        acc1_tr, _, loss_tr = train_epoch(args,
                                          train_loader,
                                          model,
                                          criterion,
                                          optimizer,
                                          epoch,
                                          logger,
                                          mixup=args.mixup)

        if epoch % args.epoch_print_freq == 0:
            acc1, acc5, loss_val = validate(args, val_loader, model, criterion, epoch, logger)

            if plotter != None:
                plotter.update(epoch, acc1_tr, acc1, loss_tr, loss_val)

            is_best = acc1 > best_acc1
            if is_best:
                best_acc1 = acc1
                best_acc5 = acc5
                if logger != None and args.verbose == True:
                    logger(f'Best accuracy (top-1 and 5): {best_acc1:.1f} {best_acc5:.1f}')
                swanlab.log({"best_acc5": best_acc5, "best_acc1": best_acc1}, step=epoch)

        if args.save_ckpt and (is_best or (epoch == args.epochs)):
            state = {
                'epoch': epoch,
                'arch': args.net_type,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc5': best_acc5,
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(args.save_dir, state, is_best)
        scheduler.step()

    return best_acc1, acc1

def distillation_loss(logits_s, logits_t, temperature):
    log_pred_student = F.log_softmax(logits_s / temperature, dim=1)
    pred_teacher = F.softmax(logits_t / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd

def evidential_criterion(logits, target):
    evidence = F.softplus(logits)
    alpha = evidence + 1
    labels_1hot = torch.zeros_like(logits).scatter_(-1, target.unsqueeze(-1), 1)
    S = torch.sum(alpha, dim=-1, keepdim=True)
    loss_ce = torch.sum(labels_1hot * (torch.digamma(S)-torch.digamma(alpha)), dim=-1).mean()
    return loss_ce

def compute_sample_weight(teacher_output, target, mode="conf", eps=1e-8, temperature=1.0):
    with torch.no_grad():
        p = torch.softmax(teacher_output, dim=1)
        pc = p.gather(1, target.view(-1,1)).squeeze(1)

        if mode == "conf":
            s = 1.0 - pc
        elif mode == "nll":
            s = -torch.log(pc.clamp_min(eps))
        elif mode == "entropy":
            s = -(p * (p.clamp_min(eps)).log()).sum(dim=1)
        elif mode == "chi":
            s = torch.sqrt((1.0 - pc).clamp_min(0.0) / pc.clamp_min(eps))
        else:
            raise ValueError(f"unknown mode: {mode}")

        if temperature != 1.0:
            s = s.pow(1.0 / temperature)

        # s = s / (s.mean().detach() + eps)

        # s = s.clamp(0.25, 4.0).detach()
    return s


def train_epoch(args,
                train_loader,
                model,
                criterion,
                optimizer,
                epoch=0,
                logger=None,
                mixup='vanilla',
                n_data=-1):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

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
            # loss = evidential_criterion(output, target)* ratio + evidential_criterion(output, target_b) * (1. - ratio)
            
        else:
            # compute output
            output = model(input)
            loss = criterion(output, target)


        # measure accuracy and record loss
        acc1, acc5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

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
    swanlab.log({"train_acc1": top1.avg, "train_acc5": top5.avg, "train_loss": losses.avg}, step=epoch)
    return top1.avg, top5.avg, losses.avg


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
    swanlab.log({"test_acc1": top1.avg, "test_acc5": top5.avg, "test_loss": losses.avg}, step=epoch)
    return top1.avg, top5.avg, losses.avg


def load_checkpoint(path, model, optimizer):
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)
        checkpoint['state_dict'] = dict(
            (key[7:], value) for (key, value) in checkpoint['state_dict'].items())
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
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
            project= "DDresnet18-Expert",
            experiment_name= args.tag,
            x_axis="epoch"
        )
    main(args, logger, args.repeat)

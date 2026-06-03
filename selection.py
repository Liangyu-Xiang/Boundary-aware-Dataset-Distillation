import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import train_models.resnet as RN
import train_models.resnet_ap as RNAP
import train_models.convnet as CN
import train_models.densenet_cifar as DN
from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from torch.utils.data import TensorDataset, DataLoader
import warnings
import shutil
from PIL import Image
from torchvision.utils import save_image
from data import load_data, MEANS, STDS
from misc.utils import AverageMeter, accuracy, get_time, Plotter
from resnet import resnet18
from tqdm import tqdm

warnings.filterwarnings("ignore")


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
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    logger(f"ImageNet directory: {args.imagenet_dir[0]}")
    _, train_loader, val_loader, nclass = load_data(args)

    # 加载教师模型
    model_teacher = define_model(args, nclass)
    teacher_path = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/imagenet10/resnet18in_resnet18imagewoof_cut/checkpoint.pth.tar"
    state_dict = torch.load(teacher_path)["state_dict"]
    model_teacher.load_state_dict(state_dict)
    model_teacher.cuda()
    model_teacher.eval()
    for p in model_teacher.parameters():
        p.requires_grad = False

    # 挑选高置信度样本
    logger("Selecting confident samples from ImageNet...")
    total_numb = select_confident_samples(
        model_teacher, train_loader, args, logger
    )

    # logger(f"Selected {len(selected_images)} confident samples.")
    # save_selected_dataset(selected_images, selected_labels, args.save_dir, logger)


@torch.no_grad()
def select_confident_samples(model, data_loader, args, logger):
    """
    按照教师模型预测准确且熵低的标准筛选ImageNet样本，并以图像形式存储。
    存储格式仿照DiT的采样脚本：
        save_dir/class_name/00000001.png
    """
    model.eval()
    device = next(model.parameters()).device
    total_selected = 0

    dataset = data_loader.dataset
    save_root = os.path.join(args.save_dir, "imagenet_confident_subset")
    os.makedirs(save_root, exist_ok=True)
    # 原始类别名称（在 find_original_classes() 中对应 ImageNet 原始目录）
    subset_class_names = dataset.classes
    assert len(subset_class_names) == dataset.nclass, \
        f"class count mismatch: {len(subset_class_names)} vs {dataset.nclass}"
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    logger(f"Start selecting confident samples... (entropy threshold={args.entropy_threshold})")

    for i, (inputs, targets) in enumerate(tqdm(data_loader)):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # 模型预测
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)
        correct = preds.eq(targets)

        # 计算熵
        entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=1)
        mask = correct & (entropy < args.entropy_threshold)

        if mask.sum() > 0:
            # 当前 batch 对应的全局样本索引范围
            batch_start = i * data_loader.batch_size
            mask_idx = mask.nonzero(as_tuple=False).view(-1).cpu().tolist()

            for j in mask_idx:
                global_idx = batch_start + j

                # 原始路径、子集类别、原始类别索引
                img_path, label = dataset.samples[global_idx]
                class_name = idx_to_class[label]
                class_dir = os.path.join(save_root, class_name)
                os.makedirs(class_dir, exist_ok=True)

                # 保存文件
                if args.copy_original:
                    dst = os.path.join(class_dir, f"{total_selected:08d}.JPEG")
                    shutil.copy2(img_path, dst)
                else:
                    img = Image.open(img_path).convert("RGB")
                    dst = os.path.join(class_dir, f"{total_selected:08d}.png")
                    img.save(dst)

                total_selected += 1

        # 日志与显存管理
        if (i + 1) % 50 == 0:
            logger(f"[{i+1}/{len(data_loader)}] Selected {total_selected} raw images...")

        del inputs, targets, outputs, probs
        torch.cuda.empty_cache()

    logger(f"✅ Done. Total {total_selected} images saved to {save_root}")
    return total_selected


def save_selected_dataset(images, labels, save_dir, logger):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "imagenet_confident_subset.pt")
    dataset = {"images": images, "labels": labels}
    torch.save(dataset, path)
    logger(f"Saved confident subset to {path}, size={len(labels)}")


if __name__ == "__main__":
    from misc.utils import Logger
    from argument import args

    os.makedirs(args.save_dir, exist_ok=True)
    logger = Logger(args.save_dir)
    logger(f"Save dir: {args.save_dir}")

    # 可调参数
    args.entropy_threshold = 0.5  # 熵阈值（越小越置信）
    args.seed = 42
    args.save_dir = "./results/imagenet_confident_subset"

    main(args, logger)

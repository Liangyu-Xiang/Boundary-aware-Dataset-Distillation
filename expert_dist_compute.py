import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from types import SimpleNamespace
from PIL import Image
from misc.utils import load_model
from data import ImageFolder


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

# ==========================================
# 1. 参数配置 (Config)
# ==========================================
class Config:
    # 路径设置
    data_path = "/data/mmc_lyxiang/dataset/ImageNet/train/"  # ImageNet 训练集路径
    save_dir = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/expert_related"
    spec = "woof"                         # 子集名称
    nclass = 10                           # 类别数量
    phase = 0                             # 阶段
    
    # 数据集相关参数
    image_size = 256
    finetune_ipc = -1                   # 如果是全量数据计算，设为 None
    num_workers = 8
    global_batch_size = 64                # 计算距离时的 Batch Size
    
    # 保存路径
    # 【动态生成保存路径】
    @property
    def save_path(self):
        return os.path.join(self.save_dir, f"expert_dist_matrix_{self.spec}.pt")

# ==========================================
# 2. 核心计算程序
# ==========================================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f">>> 使用设备: {device}")

    

    # --- A. 加载专家模型 (ResNet18) ---
    print(f">>> 正在加载专家模型: ResNet18 (Dataset: {args.spec})...")
    expert_model = load_model(
        model_name='resnet18',
        dataset=args.spec,
        pretrained=True,
        classes=range(args.nclass)
    )
    feature_dim = expert_model.fc.in_features
    expert_model.fc = nn.Identity() # 移除全连接层，直接获取特征
    expert_model.to(device).eval()

    # --- B. 准备 Transform (直接复用你的逻辑) ---
    # 计算类中心建议关闭 RandomHorizontalFlip 以获得最稳定的特征
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])

    # 这里需要定义你的 expert_transform (ResNet 标准预处理)
    expert_transform = transforms.Compose(
        [
            transforms.Resize(224 // 7 * 8, antialias=True),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # --- C. 初始化你的数据集和加载器 ---
    print(f">>> 正在初始化数据集: {args.spec} (Phase: {args.phase})")
    dataset = ImageFolder(
        args.data_path, 
        transform=transform, 
        expert_transform=expert_transform, 
        nclass=args.nclass,
        ipc=args.finetune_ipc, 
        spec=args.spec, 
        phase=args.phase,
        seed=0, 
        return_origin=True
    )

    # 距离计算不需要 DistributedSampler，单卡运行更方便
    loader = DataLoader(
        dataset,
        batch_size=args.global_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # --- D. 提取特征并累加 ---
    # 使用 ry (Relative Y) 作为索引，范围是 0 到 nclass-1
    centroids = torch.zeros(args.nclass, feature_dim, device=device)
    class_counts = torch.zeros(args.nclass, device=device)

    print(">>> 开始遍历数据集提取特征...")
    with torch.no_grad():
        for _, x_expert, ry, _ in tqdm(loader, desc="计算类中心"):
            # x_expert: 专家模型输入, ry: 子集内相对标签
            x_expert = x_expert.to(device)
            ry_t = ry.to(device).long()
            
            features = expert_model(x_expert) # [B, 512]
            
            # 这里的 ry_t 必须是 0 到 nclass-1 的连续索引
            centroids.index_add_(0, ry_t, features)
            
            # 统计每类样本数
            ones = torch.ones(ry_t.size(0), device=device)
            class_counts.index_add_(0, ry_t, ones)

    # 计算平均特征 (Centroids)
    centroids = centroids / (class_counts.unsqueeze(1) + 1e-8)

    # --- E. 计算 L2 距离矩阵 ---
    print(">>> 正在计算 L2 距离矩阵...")
    # 计算成对欧式距离
    dist_matrix = torch.cdist(centroids, centroids, p=2)

    # --- F. 保存结果 ---
    output = {
        'dist_matrix': dist_matrix.cpu(),
        'centroids': centroids.cpu(),
        'spec': args.spec,
        'class_names': dataset.classes if hasattr(dataset, 'classes') else None
    }
    torch.save(output, args.save_path)
    
    print("-" * 30)
    print(f"计算完成并保存至: {args.save_path}")
    print(f"矩阵维度: {dist_matrix.shape}")
    print("-" * 30)

if __name__ == "__main__":
    # 将 Config 类转为 namespace 对象以便使用 args.xxx 访问
    config_args = SimpleNamespace(**{k: v for k, v in Config.__dict__.items() if not k.startswith('__')})
    main(config_args)
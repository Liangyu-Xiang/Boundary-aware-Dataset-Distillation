import os
import random
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.manifold import TSNE

from misc.utils import load_model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path, _ = self.samples[index]
        return img, label, path


class ResNet18FeatureExtractor(nn.Module):
    """
    输出：
        feat   : fc 前的 512 维特征
        logits : 分类输出
    """
    def __init__(self, num_classes, spec_name, device="cuda"):
        super().__init__()

        model = load_model(
            model_name="resnet18",
            dataset=spec_name,
            pretrained=True,
            classes=range(num_classes)
        )

        self.feature_extractor = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
        )
        self.fc = model.fc
        self.device = device
        self.to(device)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        x = x.to(self.device)
        feat = self.feature_extractor(x)
        feat = torch.flatten(feat, 1)   # [B, 512]
        logits = self.fc(feat)          # [B, C]
        return feat, logits


def compute_margin(logits):
    """
    margin = top1_logit - top2_logit
    越小通常越接近决策边界
    """
    top2_vals = torch.topk(logits, k=2, dim=1).values
    margin = top2_vals[:, 0] - top2_vals[:, 1]
    return margin


@torch.no_grad()
def extract_features_and_margin(model, dataloader, domain_name):
    all_feats = []
    all_logits = []
    all_labels = []
    all_margins = []
    all_domains = []
    all_paths = []

    for images, labels, paths in dataloader:
        feats, logits = model(images)
        margins = compute_margin(logits)

        all_feats.append(feats.cpu().numpy())
        all_logits.append(logits.cpu().numpy())
        all_labels.append(labels.numpy())
        all_margins.append(margins.cpu().numpy())
        all_domains.extend([domain_name] * len(labels))
        all_paths.extend(paths)

    return {
        "features": np.concatenate(all_feats, axis=0),
        "logits": np.concatenate(all_logits, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "margins": np.concatenate(all_margins, axis=0),
        "domains": np.array(all_domains),
        "paths": np.array(all_paths),
    }


def maybe_subset(dataset, max_samples=None, seed=42):
    if max_samples is None or len(dataset) <= max_samples:
        return dataset

    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=max_samples, replace=False)
    return Subset(dataset, indices.tolist())


def get_class_colors(num_classes):
    """
    为每个类别分配固定颜色。
    当类别数 <= 20 时使用 tab20，否则使用 hsv。
    """
    if num_classes <= 20:
        cmap = plt.get_cmap("tab20", num_classes)
        colors = [cmap(i) for i in range(num_classes)]
    else:
        cmap = plt.get_cmap("hsv", num_classes)
        colors = [cmap(i) for i in range(num_classes)]
    return colors


def plot_margin_heatmap(tsne_feats, margins, domains, save_path):
    """
    主图：所有样本按 margin 着色。
    margin 越小，越接近决策边界。
    """
    plt.figure(figsize=(10, 8))

    boundary_score = -margins

    scatter = plt.scatter(
        tsne_feats[:, 0],
        tsne_feats[:, 1],
        c=boundary_score,
        s=18,
        alpha=0.8
    )

    # generated dataset 用星形 + 黑边标出
    gen_mask = (domains == "generated")
    plt.scatter(
        tsne_feats[gen_mask, 0],
        tsne_feats[gen_mask, 1],
        s=70,
        marker="*",
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        label="Ours"
    )

    cbar = plt.colorbar(scatter)
    cbar.set_label("Boundary score = -margin (larger means closer to boundary)")

    plt.title("t-SNE with Margin Heatmap")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_margin_heatmap_split(tsne_feats, margins, domains, save_path):
    """
    分开画 real / generated，便于对比两个数据集谁更偏向低 margin 区域。
    """
    real_mask = (domains == "real")
    gen_mask = (domains == "generated")

    vmin = margins.min()
    vmax = margins.max()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc1 = axes[0].scatter(
        tsne_feats[real_mask, 0],
        tsne_feats[real_mask, 1],
        c=margins[real_mask],
        s=18,
        alpha=0.8,
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title("Baseline")
    axes[0].set_xlabel("t-SNE dim 1")
    axes[0].set_ylabel("t-SNE dim 2")

    sc2 = axes[1].scatter(
        tsne_feats[gen_mask, 0],
        tsne_feats[gen_mask, 1],
        c=margins[gen_mask],
        s=28,
        alpha=0.9,
        marker="*",
        vmin=vmin,
        vmax=vmax,
        edgecolors="black",
        linewidths=0.4
    )
    axes[1].set_title("Ours")
    axes[1].set_xlabel("t-SNE dim 1")
    axes[1].set_ylabel("t-SNE dim 2")

    cbar = fig.colorbar(sc2, ax=axes.ravel().tolist())
    cbar.set_label("Margin (smaller means closer to boundary)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_margin_histogram(real_margins, gen_margins, save_path):
    """
    统计图：看 generated 是否整体更偏向低 margin。
    """
    plt.figure(figsize=(8, 6))
    plt.hist(real_margins, bins=50, alpha=0.6, density=True, label="Baseline")
    plt.hist(gen_margins, bins=50, alpha=0.6, density=True, label="Ours")
    plt.xlabel("Margin")
    plt.ylabel("Density")
    plt.title("Margin Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne_by_class(tsne_feats, labels, domains, class_names, save_path):
    """
    不同类别使用不同颜色；
    真实样本：圆点
    合成样本：星形 + 黑边 + 更大尺寸
    """
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = get_class_colors(len(class_names))

    for label in unique_labels:
        class_mask = (labels == label)
        real_mask = class_mask & (domains == "real")
        gen_mask = class_mask & (domains == "generated")

        label_name = f"Class{label + 1}"
        color = colors[label]

        # real samples
        if real_mask.sum() > 0:
            plt.scatter(
                tsne_feats[real_mask, 0],
                tsne_feats[real_mask, 1],
                s=18,
                alpha=0.65,
                color=color,
                marker="o",
                linewidths=0,
                label=f"{label_name} (baseline)"
            )

        # generated samples
        if gen_mask.sum() > 0:
            plt.scatter(
                tsne_feats[gen_mask, 0],
                tsne_feats[gen_mask, 1],
                s=90,
                alpha=0.95,
                color=color,
                marker="*",
                edgecolors="black",
                linewidths=0.7,
                label=f"{label_name} (synthetic)"
            )

    # plt.title("t-SNE by Class")
    # plt.xlabel("t-SNE dim 1")
    # plt.ylabel("t-SNE dim 2")
    plt.legend(ncol=2, fontsize="small", frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_tsne_by_class_and_domain(tsne_feats, labels, domains, class_names, save_path):
    """
    更适合论文展示的版本：
    - 颜色：类别
    - 形状：域
    - 图例拆成两部分：类别图例 + 域图例
    """
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = get_class_colors(len(class_names))

    for label in unique_labels:
        class_mask = (labels == label)
        color = colors[label]

        real_mask = class_mask & (domains == "real")
        gen_mask = class_mask & (domains == "generated")

        if real_mask.sum() > 0:
            ax.scatter(
                tsne_feats[real_mask, 0],
                tsne_feats[real_mask, 1],
                s=20,
                alpha=0.65,
                color=color,
                marker="o",
                linewidths=0,
                zorder=1
            )

        if gen_mask.sum() > 0:
            ax.scatter(
                tsne_feats[gen_mask, 0],
                tsne_feats[gen_mask, 1],
                s=95,
                alpha=0.95,
                color=color,
                marker="*",
                edgecolors="black",
                linewidths=0.8,
                zorder=3
            )

    # 类别图例
    class_handles = []
    for label in unique_labels:
        label_name = f"Class{label + 1}"
        class_handles.append(
            Line2D(
                [0], [0],
                marker="o",
                color="w",
                label=label_name,
                markerfacecolor=colors[label],
                markersize=8
            )
        )

    # 域图例
    domain_handles = [
        Line2D([0], [0], marker="o", color="gray", label="Baseline", linestyle="None", markersize=7),
        Line2D([0], [0], marker="*", color="gray", markeredgecolor="black",
               label="Ours", linestyle="None", markersize=12)
    ]

    legend1 = ax.legend(
        handles=class_handles,
        title="Class",
        loc="upper right",
        bbox_to_anchor=(1.28, 1.0),
        fontsize="small"
    )
    ax.add_artist(legend1)

    ax.legend(
        handles=domain_handles,
        title="Domain",
        loc="lower right",
        fontsize="small"
    )

    # ax.set_title("t-SNE by Class and Domain")
    # ax.set_xlabel("t-SNE dim 1")
    # ax.set_ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    set_seed(42)

    # =========================
    # 这里改成你的路径
    # =========================
    real_dataset_dir = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/dit-distillation/imagenet-10-1000-baseline-ipc50/"
    generated_dataset_dir = "/data/mmc_lyxiang/DD/MinimaxDiffusion/results/distilled_dataset/Woof/IPC50p_x_y_2_Gated+Epoch16++Temp2+Weight0.5-1+Gap0.5+Candidate4+WeightAlign+CFG1/"
    expert_spec = "woof"  # 与训练脚本中的 args.spec 保持一致

    output_dir = "./margin_heatmap_results"
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 64
    num_workers = 4
    max_samples_per_domain = 2000
    top_k_classes = 10
    # =========================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize(224 // 7 * 8, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    real_dataset = ImageFolderWithPaths(real_dataset_dir, transform=transform)
    gen_dataset = ImageFolderWithPaths(generated_dataset_dir, transform=transform)

    print("Real classes:", real_dataset.classes)
    print("Synthetic classes:", gen_dataset.classes)

    if real_dataset.classes != gen_dataset.classes:
        raise ValueError("两个数据集的类别定义不一致，请先对齐类别名和顺序。")

    num_classes = len(real_dataset.classes)

    real_dataset = maybe_subset(real_dataset, max_samples_per_domain, seed=42)
    gen_dataset = maybe_subset(gen_dataset, max_samples_per_domain, seed=43)

    real_loader = DataLoader(
        real_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    gen_loader = DataLoader(
        gen_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    model = ResNet18FeatureExtractor(
        num_classes=num_classes,
        spec_name=expert_spec,
        device=device
    )

    print("Extracting real dataset features...")
    real_info = extract_features_and_margin(model, real_loader, "real")

    print("Extracting synthetic dataset features...")
    gen_info = extract_features_and_margin(model, gen_loader, "generated")

    all_features = np.concatenate([real_info["features"], gen_info["features"]], axis=0)
    all_margins = np.concatenate([real_info["margins"], gen_info["margins"]], axis=0)
    all_domains = np.concatenate([real_info["domains"], gen_info["domains"]], axis=0)
    all_labels = np.concatenate([real_info["labels"], gen_info["labels"]], axis=0)
    class_names = real_dataset.dataset.classes if isinstance(real_dataset, Subset) else real_dataset.classes

    if top_k_classes is not None:
        top_k = min(top_k_classes, len(class_names))
        selected_indices = np.arange(top_k)
        mask = np.isin(all_labels, selected_indices)
        if mask.sum() == 0:
            raise ValueError("前 K 个类别在数据中没有样本，请检查 K 的取值。")

        all_features = all_features[mask]
        all_margins = all_margins[mask]
        all_domains = all_domains[mask]
        all_labels = all_labels[mask]

        real_label_mask = np.isin(real_info["labels"], selected_indices)
        gen_label_mask = np.isin(gen_info["labels"], selected_indices)
        real_info["margins"] = real_info["margins"][real_label_mask]
        gen_info["margins"] = gen_info["margins"][gen_label_mask]

    print("Running t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        learning_rate="auto",
        init="pca",
        random_state=42
    )
    tsne_feats = tsne.fit_transform(all_features)

    np.save(os.path.join(output_dir, "tsne_feats.npy"), tsne_feats)
    np.save(os.path.join(output_dir, "margins.npy"), all_margins)
    np.save(os.path.join(output_dir, "domains.npy"), all_domains)
    np.save(os.path.join(output_dir, "labels.npy"), all_labels)

    plot_tsne_by_class(
        tsne_feats=tsne_feats,
        labels=all_labels,
        domains=all_domains,
        class_names=class_names,
        save_path=os.path.join(output_dir, f"tsne_by_class_top{top_k_classes or 'all'}.png")
    )

    plot_tsne_by_class_and_domain(
        tsne_feats=tsne_feats,
        labels=all_labels,
        domains=all_domains,
        class_names=class_names,
        save_path=os.path.join(output_dir, f"tsne_by_class_and_domain_top{top_k_classes or 'all'}.png")
    )

    plot_margin_heatmap(
        tsne_feats=tsne_feats,
        margins=all_margins,
        domains=all_domains,
        save_path=os.path.join(output_dir, "tsne_margin_heatmap.png")
    )

    plot_margin_heatmap_split(
        tsne_feats=tsne_feats,
        margins=all_margins,
        domains=all_domains,
        save_path=os.path.join(output_dir, "tsne_margin_heatmap_split.png")
    )

    plot_margin_histogram(
        real_margins=real_info["margins"],
        gen_margins=gen_info["margins"],
        save_path=os.path.join(output_dir, "margin_histogram.png")
    )

    threshold = np.percentile(all_margins, 15)
    real_low_ratio = (real_info["margins"] <= threshold).mean()
    gen_low_ratio = (gen_info["margins"] <= threshold).mean()

    print(f"Average margin of real dataset      : {real_info['margins'].mean():.4f}")
    print(f"Average margin of synthetic dataset : {gen_info['margins'].mean():.4f}")
    print(f"Low-margin threshold (15th percentile): {threshold:.4f}")
    print(f"Low-margin ratio in real dataset      : {real_low_ratio:.4f}")
    print(f"Low-margin ratio in synthetic dataset : {gen_low_ratio:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

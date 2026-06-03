import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# ===== 参数 =====
data_dir = "/data/mmc_lyxiang/dataset/ImageNet"
num_epochs = 100
batch_size = 256
num_workers = 8
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
label_smoothing = 0.1
device = "cuda" if torch.cuda.is_available() else "cpu"

# ===== 数据增强（官方设置） =====
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = torchvision.datasets.ImageFolder(f"{data_dir}/train", transform=train_transforms)
val_dataset = torchvision.datasets.ImageFolder(f"{data_dir}/val", transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

# ===== 模型 =====
model = resnet18(weights=None)  # 从零开始训练
model = model.to(device)

# ===== 损失与优化器 =====
criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

scaler = GradScaler()

# ===== 训练循环 =====
for epoch in range(num_epochs):
    model.train()
    total, correct, running_loss = 0, 0, 0.0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", ncols=100)

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f"[Epoch {epoch+1}/{num_epochs}] Step {i+1}/{len(train_loader)} "
                  f"Loss: {running_loss/100:.3f} | Acc: {100.*correct/total:.2f}%")
            running_loss = 0.0

    scheduler.step()

    # ===== 验证 =====
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            val_correct += preds.eq(labels).sum().item()
            val_total += labels.size(0)
    print(f"Epoch [{epoch+1}/{num_epochs}] Val Acc: {100.*val_correct/val_total:.2f}%")

torch.save(model.state_dict(), "resnet18_imagenet_official_setting.pth")
print("✅ Training complete, model saved.")

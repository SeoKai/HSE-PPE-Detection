# ============================================================
# 마스크 2클래스 분류기 학습 (MobileNetV3-Small)
# 입력: datasets/v2_mask_cls/{train,val}/{mask,no_mask}/*.jpg
# 출력: weights/mask_cls.pt   (state_dict + class_names)
# 실행: python train_mask_cls.py
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets" / "v2_mask_cls"
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHT_PATH = WEIGHTS_DIR / "mask_cls.pt"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
PATIENCE = 5


def build_loaders():
    # ImageNet 정규화 (pretrained 사용하므로 맞춰야 함)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_set   = datasets.ImageFolder(DATA_DIR / "val",   transform=val_tf)

    # train 과 val 이 동일한 class 순서인지 확인
    assert train_set.classes == val_set.classes, (
        f"class mismatch: {train_set.classes} vs {val_set.classes}"
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader, train_set.classes


def build_model(num_classes: int):
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
    model = mobilenet_v3_small(weights=weights)
    # classifier 마지막 Linear 를 num_classes 로 교체
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def train():
    if not (DATA_DIR / "train").exists():
        print(f"[오류] {DATA_DIR} 없음. 먼저 prepare_mask_crops.py 실행.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("[마스크 분류기 학습]")
    print(f"  device : {device}")
    print(f"  data   : {DATA_DIR}")
    print("=" * 60)

    train_loader, val_loader, classes = build_loaders()
    print(f"  classes: {classes}")
    print(f"  train={len(train_loader.dataset)}  val={len(val_loader.dataset)}")

    model = build_model(num_classes=len(classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_acc = 0.0
    best_state = None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        bar = "█" * int(epoch / EPOCHS * 20) + "░" * (20 - int(epoch / EPOCHS * 20))
        print(
            f"[{bar}] {epoch:3d}/{EPOCHS}  "
            f"loss={train_loss:.4f}  train_acc={train_acc:.4f}  val_acc={val_acc:.4f}",
            flush=True,
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[early-stop] val_acc 개선 없음 {PATIENCE} epoch")
                break

    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "class_names": classes,
        "arch": "mobilenet_v3_small",
        "img_size": IMG_SIZE,
    }, WEIGHT_PATH)
    print("\n" + "=" * 60)
    print(f"[✓] 저장: {WEIGHT_PATH}  (best val_acc={best_acc:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    train()

# ============================================================
# 마스크 분류기 Fine-tune
# - 기존 weights/mask_cls.pt 에서 이어 학습
# - 적은 epoch(5) + 낮은 lr(1e-4) 로 약점 샘플 반영
# - 결과: weights/mask_cls.pt (덮어쓰기 전 mask_cls.prev.pt 로 백업)
# ============================================================

import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_small
from pathlib import Path

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "datasets" / "v2_mask_cls"
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHT_PATH = WEIGHTS_DIR / "mask_cls.pt"
BACKUP_PATH = WEIGHTS_DIR / "mask_cls.prev.pt"

IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-4          # fine-tune: base보다 10배 작게
PATIENCE = 3


def build_loaders():
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
    assert train_set.classes == val_set.classes, \
        f"class mismatch: {train_set.classes} vs {val_set.classes}"

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader, train_set.classes


def build_model(num_classes: int, device):
    # weights=None: 어차피 아래에서 ckpt 로드
    model = mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total if total else 0.0


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"{WEIGHT_PATH} 없음 — 먼저 train_mask_cls.py 를 실행하세요.")

    # 백업
    shutil.copy(WEIGHT_PATH, BACKUP_PATH)
    print(f"[backup] {WEIGHT_PATH.name} → {BACKUP_PATH.name}")

    ckpt = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]

    train_loader, val_loader, folder_classes = build_loaders()
    assert folder_classes == class_names, \
        f"폴더 클래스 순서 불일치: {folder_classes} vs {class_names}"

    model = build_model(num_classes=len(class_names), device=device)
    model.load_state_dict(ckpt["state_dict"])

    # 기준 val_acc
    base_acc = evaluate(model, val_loader, device)
    print(f"[base val_acc] {base_acc:.4f}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_acc = base_acc
    best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
    bad = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total, running = 0, 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optim.step()
            running += loss.item() * y.size(0)
            total += y.size(0)
        train_loss = running / max(total, 1)

        val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch:02d}] train_loss={train_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"[early stop] epoch {epoch}")
                break

    print(f"\n[best val_acc] {best_acc:.4f}  (base였음 {base_acc:.4f})")
    if best_acc < base_acc:
        print("⚠️  fine-tune 성능이 기존보다 낮음 → 가중치 덮어쓰지 않음. 백업 파일 유지.")
        return

    torch.save({"state_dict": best_state,
                "class_names": class_names,
                "img_size": IMG_SIZE},
               WEIGHT_PATH)
    print(f"[saved] {WEIGHT_PATH}")


if __name__ == "__main__":
    main()

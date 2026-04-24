# ============================================================
# 헬멧 전용 YOLOv8 학습 스크립트 (2클래스: helmet / no_helmet)
# 실행: python train_helmet.py
# 출력: weights/helmet_best.pt
# ============================================================

import shutil
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
DATA_YAML = BASE_DIR / "datasets" / "v2_helmet" / "data.yaml"
WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHT_NAME = "helmet_best.pt"

CLASS_NAMES = ["helmet", "no_helmet"]


def on_train_epoch_end(trainer):
    m = trainer.metrics
    e = trainer.epoch + 1
    total = trainer.epochs
    box_l = m.get("train/box_loss", 0)
    cls_l = m.get("train/cls_loss", 0)
    p     = m.get("metrics/precision(B)", 0)
    r     = m.get("metrics/recall(B)", 0)
    map50 = m.get("metrics/mAP50(B)", 0)
    bar   = "█" * int(e / total * 20) + "░" * (20 - int(e / total * 20))
    print(
        f"\n[{bar}] {e:3d}/{total}  box={box_l:.3f}  cls={cls_l:.3f}  "
        f"P={p:.3f}  R={r:.3f}  mAP50={map50:.3f}",
        flush=True,
    )


def on_train_end(trainer):
    m = trainer.metrics
    print("\n" + "=" * 60)
    print("[헬멧 학습 완료]")
    print(f"  Precision : {m.get('metrics/precision(B)', 0):.4f}")
    print(f"  Recall    : {m.get('metrics/recall(B)', 0):.4f}")
    print(f"  mAP@0.5   : {m.get('metrics/mAP50(B)', 0):.4f}")
    print(f"  mAP@0.5:95: {m.get('metrics/mAP50-95(B)', 0):.4f}")
    print("=" * 60)


def train():
    """
    헬멧 전용 2클래스 학습.
    - imgsz=960 (RTX 4060 여유 있음, 소형 헬멧도 정확하게 감지)
    - Adam, lr0=1e-4, mosaic=1.0, mixup=0.1, label_smoothing=0.05
    """
    if not DATA_YAML.exists():
        print(f"[오류] {DATA_YAML} 없음. 먼저 prepare_helmet_dataset.py 를 실행하세요.")
        return

    device = 0 if torch.cuda.is_available() else "cpu"
    model_name = "yolov8n.pt" if device == "cpu" else "yolov8s.pt"

    print("=" * 60)
    print("[헬멧 전용 YOLO 학습]")
    print(f"  data   : {DATA_YAML}")
    print(f"  device : {'GPU (CUDA)' if device == 0 else 'CPU'}")
    print(f"  model  : {model_name}")
    print("=" * 60)

    model = YOLO(model_name)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    results = model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=24,              # 640 해상도 + 4060
        patience=8,
        device=device,
        workers=0,             # Windows 호환

        optimizer="Adam",
        lr0=1e-4,

        mosaic=1.0,
        mixup=0.1,
        scale=0.5,
        fliplr=0.5,
        degrees=10.0,
        shear=2.0,
        close_mosaic=10,

        box=10.0,
        cls=1.5,
        label_smoothing=0.05,

        project="runs",
        name="helmet_2cls",
        exist_ok=True,
        verbose=True,
    )

    # Ultralytics는 절대경로(~/runs/detect/...)로 저장할 수 있으므로
    # trainer.save_dir에서 실제 위치를 가져온다.
    WEIGHTS_DIR.mkdir(exist_ok=True)
    dest = WEIGHTS_DIR / WEIGHT_NAME
    best_src = None
    try:
        save_dir = Path(model.trainer.save_dir)
        cand = save_dir / "weights" / "best.pt"
        if cand.exists():
            best_src = cand
    except Exception:
        pass
    if best_src is None:
        # fallback 후보들
        for c in (
            Path("runs") / "helmet_2cls" / "weights" / "best.pt",
            Path.home() / "runs" / "detect" / "runs" / "helmet_2cls" / "weights" / "best.pt",
        ):
            if c.exists():
                best_src = c
                break
    if best_src is not None:
        shutil.copy(best_src, dest)
        print(f"\n[✓] 최종 가중치 복사: {best_src} -> {dest}")
    else:
        print(f"[!] best.pt 미발견")


if __name__ == "__main__":
    train()

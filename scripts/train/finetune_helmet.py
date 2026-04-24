# ============================================================
# 헬멧 탐지기 Fine-tune
# - Roboflow export 폴더의 이미지/라벨을 datasets/v2_helmet/train 에 병합
# - 기존 weights/helmet_best.pt 에서 이어 학습(적은 epoch, 작은 lr)
# - 결과: weights/helmet_best.pt (덮어쓰기 전 helmet_best.prev.pt 로 백업)
#
# 사용:
#   .venv/Scripts/python.exe finetune_helmet.py --merge
#   .venv/Scripts/python.exe finetune_helmet.py           # merge 는 한 번만 필요
# ============================================================

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent
ROBOFLOW_DIR = Path(r"C:\Users\momo\Downloads\Helmet.v1i.yolov8")
DATA_YAML = BASE_DIR / "datasets" / "v2_helmet" / "data.yaml"
TRAIN_IMG = BASE_DIR / "datasets" / "v2_helmet" / "train" / "images"
TRAIN_LBL = BASE_DIR / "datasets" / "v2_helmet" / "train" / "labels"

WEIGHTS_DIR = BASE_DIR / "weights"
WEIGHT_PATH = WEIGHTS_DIR / "helmet_best.pt"
BACKUP_PATH = WEIGHTS_DIR / "helmet_best.prev.pt"


def merge_roboflow():
    """Roboflow train/images + train/labels 를 기존 v2_helmet/train 에 복사"""
    src_img = ROBOFLOW_DIR / "train" / "images"
    src_lbl = ROBOFLOW_DIR / "train" / "labels"
    if not src_img.exists() or not src_lbl.exists():
        raise FileNotFoundError(f"Roboflow export 경로가 이상합니다: {ROBOFLOW_DIR}")

    TRAIN_IMG.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png"}
    n_img = n_lbl = 0
    for p in src_img.iterdir():
        if p.suffix.lower() in img_exts:
            shutil.copy2(p, TRAIN_IMG / p.name)
            n_img += 1
    for p in src_lbl.iterdir():
        if p.suffix.lower() == ".txt":
            shutil.copy2(p, TRAIN_LBL / p.name)
            n_lbl += 1
    print(f"[merge] images {n_img} + labels {n_lbl} 복사 완료")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", action="store_true", help="Roboflow export 병합 후 학습")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4, help="fine-tune 용 작은 lr")
    args = ap.parse_args()

    if args.merge:
        merge_roboflow()

    if not WEIGHT_PATH.exists():
        raise FileNotFoundError(f"{WEIGHT_PATH} 없음 — 먼저 train_helmet.py 를 실행하세요.")

    # 백업
    shutil.copy2(WEIGHT_PATH, BACKUP_PATH)
    print(f"[backup] {WEIGHT_PATH.name} → {BACKUP_PATH.name}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # 기존 best 에서 이어 학습
    model = YOLO(str(WEIGHT_PATH))

    results = model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=6,
        device=device,
        workers=0,
        optimizer="AdamW",
        lr0=args.lr,
        # 데이터 적은 fine-tune 이므로 강한 증강 유지
        mosaic=1.0,
        mixup=0.1,
        scale=0.5,
        fliplr=0.5,
        degrees=10.0,
        shear=2.0,
        close_mosaic=5,   # 끝 5 epoch 은 mosaic 끄고 현실적 학습
        box=10.0,
        cls=1.5,
        label_smoothing=0.05,
        project="runs",
        name="helmet_ft",
        exist_ok=True,
        verbose=True,
    )

    # fine-tune 결과 best.pt 를 weights/helmet_best.pt 로 복사
    try:
        save_dir = Path(model.trainer.save_dir)
    except Exception:
        save_dir = BASE_DIR / "runs" / "helmet_ft"
    new_best = save_dir / "weights" / "best.pt"
    if not new_best.exists():
        # 대안 경로 (ultralytics 기본값)
        alt = Path.home() / "runs" / "detect" / "runs" / "helmet_ft" / "weights" / "best.pt"
        if alt.exists():
            new_best = alt

    if not new_best.exists():
        print(f"⚠️  best.pt 를 찾지 못했습니다 (탐색: {save_dir}). 수동 복사 필요.")
        return

    shutil.copy2(new_best, WEIGHT_PATH)
    print(f"[saved] {new_best} → {WEIGHT_PATH}")
    print("\n[요약] 학습 지표는 위 로그 또는 runs/helmet_ft/results.csv 참조")


if __name__ == "__main__":
    main()

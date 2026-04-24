# ============================================================
# 헬멧 YOLO Fine-tune (웹캠 클로즈업 도메인 보강)
# - Base : weights/helmet_best.pt (mAP50 88.2% baseline)
# - Data : datasets/webcam_helmet/data.yaml (146 train / 27 valid)
# - 전략 : 낮은 lr, 적은 epochs, mosaic/mixup 끄기 (클로즈업 왜곡 방지)
# - 출력 : runs/helmet_webcam_ft/weights/best.pt
# - 베이스라인 가중치는 절대 덮어쓰지 않음. 수동으로 비교 후 복사 결정.
# ============================================================

import shutil
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).parent
BASE_WEIGHTS = ROOT / "weights" / "helmet_best.pt"
DATA_YAML = ROOT / "datasets" / "webcam_helmet" / "data.yaml"
RUN_NAME = "helmet_webcam_ft"
PROJECT_DIR = ROOT / "runs"


def main():
    assert BASE_WEIGHTS.exists(), f"not found: {BASE_WEIGHTS}"
    assert DATA_YAML.exists(), f"not found: {DATA_YAML}"

    print(f"[base]  {BASE_WEIGHTS}")
    print(f"[data]  {DATA_YAML}")

    model = YOLO(str(BASE_WEIGHTS))

    model.train(
        data=str(DATA_YAML),
        epochs=25,
        imgsz=640,
        batch=16,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        exist_ok=True,
        # -- 학습률: 베이스 가중치를 깨지 않도록 매우 작게
        lr0=5e-5,
        lrf=0.01,
        warmup_epochs=1.0,
        optimizer="AdamW",
        weight_decay=0.0005,
        # -- augmentation: 클로즈업에 해로운 것 제거
        mosaic=0.0,       # 4장 합성 X (클로즈업 왜곡)
        mixup=0.0,
        copy_paste=0.0,
        degrees=5.0,      # 회전 약간
        translate=0.05,
        scale=0.2,        # 스케일 변화 약간
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,       # 좌우 반전은 유지
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.3,
        # -- 기타
        patience=10,
        save=True,
        verbose=True,
        plots=True,
        workers=0,        # Windows + Python 3.14 dataloader 크래시 회피
        cache="ram",      # 146장이라 RAM에 올려도 됨 (속도도 빠름)
    )

    # 결과 위치
    best = PROJECT_DIR / RUN_NAME / "weights" / "best.pt"
    last = PROJECT_DIR / RUN_NAME / "weights" / "last.pt"
    print("\n=== fine-tune 완료 ===")
    print(f"best: {best}")
    print(f"last: {last}")
    print("\n※ weights/helmet_best.pt 는 덮어쓰지 않았음.")
    print("  카메라 테스트로 비교 후 수동으로 복사 결정:")
    print(f"    cp '{best}' weights/helmet_best.pt")


if __name__ == "__main__":
    main()

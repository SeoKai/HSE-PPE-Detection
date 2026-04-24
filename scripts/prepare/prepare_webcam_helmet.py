# ============================================================
# 웹캠 헬멧 데이터셋 준비
# - Source: C:\Users\momo\Downloads\Helmet.yolov8\train (images + labels)
# - Dest  : datasets/webcam_helmet/{train,valid,test}/{images,labels}
# - Split : 80 / 15 / 5 (seed=42)
# - 기존 raw 폴더는 그대로 둠 (원본 백업 용도)
# ============================================================

import random
import shutil
from pathlib import Path

SRC = Path(r"C:\Users\momo\Downloads\Helmet.yolov8\train")
DST = Path(__file__).parent / "datasets" / "webcam_helmet"

SPLIT = {"train": 0.80, "valid": 0.15, "test": 0.05}
SEED = 42

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    src_img = SRC / "images"
    src_lbl = SRC / "labels"
    assert src_img.exists(), f"not found: {src_img}"
    assert src_lbl.exists(), f"not found: {src_lbl}"

    # 기존 split 폴더 정리
    for split in SPLIT:
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)
        # 비우기
        for p in (DST / split / "images").iterdir():
            if p.is_file():
                p.unlink()
        for p in (DST / split / "labels").iterdir():
            if p.is_file():
                p.unlink()

    images = sorted([p for p in src_img.iterdir() if p.suffix.lower() in IMG_EXTS])
    print(f"source images: {len(images)}")

    random.seed(SEED)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT["train"])
    n_valid = int(n * SPLIT["valid"])
    buckets = {
        "train": images[:n_train],
        "valid": images[n_train : n_train + n_valid],
        "test": images[n_train + n_valid :],
    }

    counts = {}
    for split, files in buckets.items():
        cnt_img = cnt_lbl = 0
        for img in files:
            lbl = src_lbl / (img.stem + ".txt")
            shutil.copy2(img, DST / split / "images" / img.name)
            cnt_img += 1
            if lbl.exists():
                shutil.copy2(lbl, DST / split / "labels" / lbl.name)
                cnt_lbl += 1
            else:
                # 빈 라벨 파일 생성 (배경 이미지)
                (DST / split / "labels" / (img.stem + ".txt")).write_text("", encoding="utf-8")
                cnt_lbl += 1
        counts[split] = (cnt_img, cnt_lbl)

    # data.yaml 생성 (절대경로)
    data_yaml = DST / "data.yaml"
    data_yaml.write_text(
        "names:\n"
        "- helmet\n"
        "- no_helmet\n"
        "nc: 2\n"
        f"train: {(DST / 'train' / 'images').resolve()}\n"
        f"val: {(DST / 'valid' / 'images').resolve()}\n"
        f"test: {(DST / 'test' / 'images').resolve()}\n",
        encoding="utf-8",
    )

    print("=== split result ===")
    for split, (ci, cl) in counts.items():
        print(f"  {split}: {ci} images / {cl} labels")
    print(f"\ndata.yaml: {data_yaml}")


if __name__ == "__main__":
    main()

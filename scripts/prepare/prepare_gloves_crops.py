# ============================================================
# 장갑 분류기용 크롭 추출
# - datasets/merged 에서 Gloves(ID=0) / NO-Gloves(ID=5) 라벨을 찾아
#   해당 바운딩 박스를 40% 패딩해서 정사각형으로 크롭 (손가락 끝까지 포함)
# - 160x160 로 리사이즈하여 ImageFolder 구조로 저장
#
# 출력:
#   datasets/v2_gloves_cls/train/{gloves,no_gloves}/
#   datasets/v2_gloves_cls/val/{gloves,no_gloves}/
# 실행: python prepare_gloves_crops.py
# ============================================================

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

BASE_DIR = Path.cwd()
MERGED_DIR = BASE_DIR / "datasets" / "merged"
OUT_DIR = BASE_DIR / "datasets" / "v2_gloves_cls"


def imread_unicode(path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def imwrite_unicode(path, img, quality=90):
    try:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        buf.tofile(str(path))
        return True
    except Exception:
        return False

GLOVE_CLASS_IDS = {
    0: "gloves",
    5: "no_gloves",
}

CROP_SIZE = 160
PAD_RATIO = 0.40   # 손은 장갑 넘어 손목/팔이 붙어있어야 구분이 쉬움 -> 패딩 크게
VAL_RATIO = 0.15

random.seed(42)


def crop_with_padding(img, cx, cy, w, h, pad_ratio=PAD_RATIO):
    H, W = img.shape[:2]
    box_w = w * W
    box_h = h * H
    cx_px = cx * W
    cy_px = cy * H
    side = max(box_w, box_h) * (1 + pad_ratio * 2)
    half = side / 2
    x1 = max(0, int(round(cx_px - half)))
    y1 = max(0, int(round(cy_px - half)))
    x2 = min(W, int(round(cx_px + half)))
    y2 = min(H, int(round(cy_px + half)))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    return img[y1:y2, x1:x2]


def process_split(split_name: str, out_split_name: str):
    labels_dir = MERGED_DIR / split_name / "labels"
    images_dir = MERGED_DIR / split_name / "images"
    if not labels_dir.exists():
        return {v: 0 for v in GLOVE_CLASS_IDS.values()}

    counts = {v: 0 for v in GLOVE_CLASS_IDS.values()}

    for lbl_file in labels_dir.glob("*.txt"):
        stem = lbl_file.stem
        img_path = None
        for ext in (".jpg", ".jpeg", ".png"):
            p = images_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue

        img = imread_unicode(img_path)
        if img is None:
            continue

        with open(lbl_file, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                except ValueError:
                    continue
                if cid not in GLOVE_CLASS_IDS:
                    continue

                crop = crop_with_padding(img, cx, cy, w, h)
                if crop is None:
                    continue
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                  interpolation=cv2.INTER_AREA)

                cls_name = GLOVE_CLASS_IDS[cid]
                out_dir = OUT_DIR / out_split_name / cls_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{stem}_{i}.jpg"
                imwrite_unicode(out_path, crop, 90)
                counts[cls_name] += 1

    return counts


def split_train_val():
    for cls_name in GLOVE_CLASS_IDS.values():
        src = OUT_DIR / "train" / cls_name
        dst = OUT_DIR / "val" / cls_name
        if not src.exists():
            continue
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob("*.jpg"))
        random.shuffle(files)
        n_val = int(len(files) * VAL_RATIO)
        for f in files[:n_val]:
            shutil.move(str(f), str(dst / f.name))


def main():
    if OUT_DIR.exists():
        print(f"[!] 기존 {OUT_DIR} 삭제")
        shutil.rmtree(OUT_DIR)

    print("=" * 60)
    print("[장갑 분류기 크롭 추출]")
    print("=" * 60)

    print("[train] merged/train 에서 추출")
    c = process_split("train", "train")
    for k, v in c.items():
        print(f"  {k:10s}: {v}")

    for split in ("valid", "test"):
        print(f"[{split}] merged/{split} 에서 추출 -> val")
        c = process_split(split, "val")
        for k, v in c.items():
            print(f"  {k:10s}: {v}")

    print("\n[split] train 15% -> val")
    split_train_val()

    print("\n" + "=" * 60)
    print("[최종 통계]")
    for split in ("train", "val"):
        print(f"  [{split}]")
        for cls_name in GLOVE_CLASS_IDS.values():
            d = OUT_DIR / split / cls_name
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"    {cls_name:10s}: {n}")
    print("=" * 60)


if __name__ == "__main__":
    main()

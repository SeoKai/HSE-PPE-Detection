# ============================================================
# 마스크 분류기용 크롭 추출
# - datasets/merged 에서 Mask(ID=2) / NO-Mask(ID=4) 라벨을 찾아
#   해당 바운딩 박스를 20% 패딩해서 정사각형으로 크롭
# - 224x224 로 리사이즈하여 ImageFolder 구조로 저장
#
# 출력:
#   datasets/v2_mask_cls/train/mask/
#   datasets/v2_mask_cls/train/no_mask/
#   datasets/v2_mask_cls/val/{mask,no_mask}/
# 실행: python prepare_mask_crops.py
# ============================================================

import cv2
import numpy as np
import random
import shutil
from pathlib import Path

# cwd 기반으로 경로 설정 (한글 경로 mojibake 회피)
BASE_DIR = Path.cwd()
MERGED_DIR = BASE_DIR / "datasets" / "merged"
OUT_DIR = BASE_DIR / "datasets" / "v2_mask_cls"


def imread_unicode(path):
    """Windows에서 한글 경로 대응 imread"""
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

# merged 클래스 ID
MASK_CLASS_IDS = {
    2: "mask",
    4: "no_mask",
}

CROP_SIZE = 224
PAD_RATIO = 0.20   # 박스 주변 20% 여유
VAL_RATIO = 0.15   # train 중 15%를 val 로 분리

random.seed(42)


def crop_with_padding(img, cx, cy, w, h, pad_ratio=PAD_RATIO):
    """
    YOLO 정규화 좌표 (cx,cy,w,h, 0~1) 를 픽셀로 변환하고
    박스를 정사각형으로 확장 + 패딩해서 크롭한다.
    이미지 경계는 클램핑.
    """
    H, W = img.shape[:2]
    box_w = w * W
    box_h = h * H
    cx_px = cx * W
    cy_px = cy * H

    # 더 큰 변 기준 정사각형 + 패딩
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
    """
    merged/{split_name} 에서 mask/no_mask 크롭을 추출해
    v2_mask_cls/{out_split_name}/{class}/ 에 저장한다.
    반환: {class: count}
    """
    labels_dir = MERGED_DIR / split_name / "labels"
    images_dir = MERGED_DIR / split_name / "images"
    if not labels_dir.exists():
        return {v: 0 for v in MASK_CLASS_IDS.values()}

    counts = {v: 0 for v in MASK_CLASS_IDS.values()}

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
                if cid not in MASK_CLASS_IDS:
                    continue

                crop = crop_with_padding(img, cx, cy, w, h)
                if crop is None:
                    continue
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                  interpolation=cv2.INTER_AREA)

                cls_name = MASK_CLASS_IDS[cid]
                out_dir = OUT_DIR / out_split_name / cls_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{stem}_{i}.jpg"
                imwrite_unicode(out_path, crop, 90)
                counts[cls_name] += 1

    return counts


def split_train_val():
    """
    v2_mask_cls/train/ 의 일부(VAL_RATIO)를 val/ 로 이동.
    process_split 직후 호출.
    """
    for cls_name in MASK_CLASS_IDS.values():
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
    print("[마스크 분류기 크롭 추출]")
    print("=" * 60)

    total = {v: 0 for v in MASK_CLASS_IDS.values()}

    # merged/train -> v2_mask_cls/train
    print("[train] merged/train 에서 추출")
    c = process_split("train", "train")
    for k, v in c.items():
        print(f"  {k:10s}: {v}")
        total[k] += v

    # merged/valid + test -> v2_mask_cls/val 일부에 합침 (더 많은 val 확보)
    # 우선 train 쪽으로 넣고 아래에서 split
    # valid/test 는 그대로 val 에 흡수시킨다.
    for split in ("valid", "test"):
        print(f"[{split}] merged/{split} 에서 추출 -> val")
        c = process_split(split, "val")
        for k, v in c.items():
            print(f"  {k:10s}: {v}")
            total[k] += v

    # train 중 일부를 val 로 분리 (val 이 너무 적을 때 보강)
    print("\n[split] train 15% -> val")
    split_train_val()

    # 최종 통계
    print("\n" + "=" * 60)
    print("[최종 통계]")
    for split in ("train", "val"):
        print(f"  [{split}]")
        for cls_name in MASK_CLASS_IDS.values():
            d = OUT_DIR / split / cls_name
            n = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"    {cls_name:10s}: {n}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# ============================================================
# 데이터셋 품질 필터링 스크립트
# - 대상: datasets/merged (train/valid/test)
# - 제거 후보:
#   1) orphan: 라벨 없는 이미지 / 이미지 없는 라벨
#   2) 깨진 이미지 (열기 실패)
#   3) 저해상도 (min(h,w) < MIN_SIDE)
#   4) 헬멧 관련 클래스(Hardhat/NO-Hardhat) bbox 면적이 MIN_AREA_RATIO 미만인 경우
#      → 이미지의 다른 라벨이 모두 없어지면 이미지 전체 삭제
#
# 기본 DRY RUN. 실제 삭제는 --apply 플래그.
# 사용법:
#   python filter_dataset.py              # 스캔만
#   python filter_dataset.py --apply      # 실제 삭제
# ============================================================

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent / "datasets" / "merged"
SPLITS = ["train", "valid", "test"]

# merged data.yaml 기준 클래스 인덱스
# 0: Gloves, 1: Hardhat, 2: Mask, 3: NO-Hardhat, 4: NO-Mask, 5: NO-Gloves
HELMET_CLASS_IDS = {1, 3}

MIN_SIDE = 320           # 최소 가로/세로 (이보다 작으면 저해상도)
MIN_AREA_RATIO = 0.001   # 헬멧 bbox 면적 / 이미지 면적 (0.1%)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def imread_unicode(path: Path):
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def parse_label(path: Path):
    """YOLO txt → list[(cls, xc, yc, w, h)]"""
    out = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            out.append((cls, xc, yc, w, h))
    except Exception:
        return None
    return out


def scan_split(split: str):
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    if not img_dir.exists():
        print(f"[skip] {img_dir} 없음")
        return [], [], []

    images = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    labels = {p.stem: p for p in lbl_dir.iterdir() if p.suffix.lower() == ".txt"} if lbl_dir.exists() else {}

    to_delete_imgs = []   # (path, reason)
    to_delete_lbls = []   # (path, reason)
    to_rewrite_lbls = []  # (path, new_lines, removed_count)

    img_stems = set()
    for img in images:
        img_stems.add(img.stem)
        lbl = labels.get(img.stem)

        # 1) orphan image
        if lbl is None:
            to_delete_imgs.append((img, "orphan(no label)"))
            continue

        # 2) 깨진 이미지
        arr = imread_unicode(img)
        if arr is None:
            to_delete_imgs.append((img, "corrupted"))
            to_delete_lbls.append((lbl, "pair of corrupted image"))
            continue

        h, w = arr.shape[:2]
        if min(h, w) < MIN_SIDE:
            to_delete_imgs.append((img, f"low-res {w}x{h}"))
            to_delete_lbls.append((lbl, "pair of low-res image"))
            continue

        # 3) 라벨 파싱
        rows = parse_label(lbl)
        if rows is None:
            to_delete_imgs.append((img, "label parse error"))
            to_delete_lbls.append((lbl, "parse error"))
            continue

        if len(rows) == 0:
            # 빈 라벨 = 배경 이미지. 유지(negative sample로 유용).
            continue

        # 4) 너무 작은 헬멧 bbox만 제거, 나머지는 유지
        kept = []
        removed = 0
        for cls, xc, yc, bw, bh in rows:
            if cls in HELMET_CLASS_IDS:
                area = bw * bh  # 이미 정규화된 비율
                if area < MIN_AREA_RATIO:
                    removed += 1
                    continue
            kept.append((cls, xc, yc, bw, bh))

        if removed > 0:
            if len(kept) == 0:
                # 이 이미지의 모든 박스가 너무 작은 헬멧이었음 → 이미지+라벨 통째로 삭제
                to_delete_imgs.append((img, f"all {removed} helmet boxes too small"))
                to_delete_lbls.append((lbl, "all boxes removed"))
            else:
                new_lines = [f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for (c, x, y, w, h) in kept]
                to_rewrite_lbls.append((lbl, new_lines, removed))

    # orphan labels (이미지가 없는 라벨)
    for stem, lbl in labels.items():
        if stem not in img_stems:
            to_delete_lbls.append((lbl, "orphan(no image)"))

    return to_delete_imgs, to_delete_lbls, to_rewrite_lbls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="실제 삭제/수정 수행")
    ap.add_argument("--verbose", action="store_true", help="모든 항목 출력")
    args = ap.parse_args()

    print(f"=== dataset filter ({'APPLY' if args.apply else 'DRY-RUN'}) ===")
    print(f"root: {ROOT}")
    print(f"min_side={MIN_SIDE}, helmet min_area_ratio={MIN_AREA_RATIO}")
    print()

    total_img_del = 0
    total_lbl_del = 0
    total_lbl_fix = 0
    reasons = {}

    for split in SPLITS:
        print(f"--- [{split}] ---")
        dels_img, dels_lbl, rewrites = scan_split(split)
        print(f"  images to delete : {len(dels_img)}")
        print(f"  labels to delete : {len(dels_lbl)}")
        print(f"  labels to rewrite: {len(rewrites)}")

        for _, r in dels_img:
            reasons[r] = reasons.get(r, 0) + 1

        if args.verbose:
            for p, r in dels_img[:20]:
                print(f"    DEL IMG {p.name}  ({r})")
            for p, r in dels_lbl[:20]:
                print(f"    DEL LBL {p.name}  ({r})")
            for p, _, n in rewrites[:20]:
                print(f"    FIX LBL {p.name}  (-{n} boxes)")
            if len(dels_img) + len(dels_lbl) + len(rewrites) > 60:
                print("    ... (생략)")

        if args.apply:
            for p, _ in dels_img:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"  [err] {p}: {e}")
            for p, _ in dels_lbl:
                try:
                    if p.exists():
                        p.unlink()
                except Exception as e:
                    print(f"  [err] {p}: {e}")
            for p, lines, _ in rewrites:
                try:
                    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
                except Exception as e:
                    print(f"  [err] {p}: {e}")

        total_img_del += len(dels_img)
        total_lbl_del += len(dels_lbl)
        total_lbl_fix += len(rewrites)
        print()

    print("=== summary ===")
    print(f"images to delete : {total_img_del}")
    print(f"labels to delete : {total_lbl_del}")
    print(f"labels to rewrite: {total_lbl_fix}")
    print("reason breakdown :")
    for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {n:6d}  {r}")
    if not args.apply:
        print("\n(dry-run) 실제 적용하려면 --apply 플래그 추가")


if __name__ == "__main__":
    main()

# ============================================================
# 헬멧 전용 2클래스 YOLO 데이터셋 준비
# - datasets/merged 에서 Hardhat(ID=1) / NO-Hardhat(ID=3) 라벨만 추출
# - datasets/tmp_check_helmets (head=0, helmet=1) 를 helmet/no_helmet 으로 리매핑 병합
# 출력:
#   datasets/v2_helmet/{train,valid,test}/{images,labels}
#   datasets/v2_helmet/data.yaml   (클래스: helmet, no_helmet)
# 실행: python prepare_helmet_dataset.py
# ============================================================

import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
MERGED_DIR = BASE_DIR / "datasets" / "merged"
CHECK_HELMETS_DIR = BASE_DIR / "datasets" / "tmp_check_helmets"
OUT_DIR = BASE_DIR / "datasets" / "v2_helmet"

# 신규 클래스 정의 (요청: 소문자)
#   0 = helmet
#   1 = no_helmet
NEW_CLASS_NAMES = ["helmet", "no_helmet"]

# merged 데이터셋의 클래스 매핑
# (prepare_dataset.py 기준: 0=Gloves, 1=Hardhat, 2=Mask, 3=NO-Hardhat, 4=NO-Mask, 5=NO-Gloves)
MERGED_CLASS_MAP = {
    1: 0,  # Hardhat    -> helmet
    3: 1,  # NO-Hardhat -> no_helmet
}

# Hard Hat Workers 데이터셋 (tmp_check_helmets) 클래스 매핑
# data.yaml: 0=head, 1=helmet
CHECK_CLASS_MAP = {
    0: 1,  # head   -> no_helmet
    1: 0,  # helmet -> helmet
}


def remap_labels(src_labels_dir: Path, src_images_dir: Path,
                 dst_labels_dir: Path, dst_images_dir: Path,
                 class_map: dict, prefix: str = ""):
    """
    라벨 파일을 class_map 기준으로 리매핑하여 복사.
    class_map 에 없는 클래스의 바운딩 박스는 버린다.
    대응하는 이미지도 같이 복사한다.
    라벨이 비어버린 파일은 스킵한다 (YOLO는 빈 라벨을 negative 로 취급).
    """
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    dst_images_dir.mkdir(parents=True, exist_ok=True)

    kept_files = 0
    kept_boxes = 0
    skipped_files = 0

    for label_file in src_labels_dir.glob("*.txt"):
        new_lines = []
        with open(label_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                except ValueError:
                    continue
                if cls_id in class_map:
                    new_id = class_map[cls_id]
                    new_lines.append(f"{new_id} {' '.join(parts[1:])}")
                    kept_boxes += 1

        if not new_lines:
            skipped_files += 1
            continue

        # 대응 이미지 찾기 (jpg/jpeg/png)
        stem = label_file.stem
        img_src = None
        for ext in (".jpg", ".jpeg", ".png"):
            candidate = src_images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_src = candidate
                break
        if img_src is None:
            skipped_files += 1
            continue

        new_stem = f"{prefix}{stem}" if prefix else stem
        (dst_labels_dir / f"{new_stem}.txt").write_text(
            "\n".join(new_lines) + "\n", encoding="utf-8"
        )
        shutil.copy2(img_src, dst_images_dir / f"{new_stem}{img_src.suffix}")
        kept_files += 1

    return kept_files, kept_boxes, skipped_files


def process_merged_split(split: str):
    """merged/{split} 에서 헬멧 관련 라벨만 추출"""
    src_labels = MERGED_DIR / split / "labels"
    src_images = MERGED_DIR / split / "images"
    dst_labels = OUT_DIR / split / "labels"
    dst_images = OUT_DIR / split / "images"

    if not src_labels.exists():
        print(f"  [skip] {src_labels} 없음")
        return

    files, boxes, skipped = remap_labels(
        src_labels, src_images, dst_labels, dst_images,
        MERGED_CLASS_MAP, prefix=""
    )
    print(f"  [merged/{split}] 파일={files}  박스={boxes}  스킵={skipped}")


def process_check_helmets():
    """
    tmp_check_helmets (train 만 존재) 를 v2_helmet/train 에 병합.
    head -> no_helmet, helmet -> helmet 매핑.
    """
    src_labels = CHECK_HELMETS_DIR / "train" / "labels"
    src_images = CHECK_HELMETS_DIR / "train" / "images"
    dst_labels = OUT_DIR / "train" / "labels"
    dst_images = OUT_DIR / "train" / "images"

    if not src_labels.exists():
        print(f"  [skip] {src_labels} 없음")
        return

    files, boxes, skipped = remap_labels(
        src_labels, src_images, dst_labels, dst_images,
        CHECK_CLASS_MAP, prefix="hh_"  # 파일명 중복 방지
    )
    print(f"  [hard_hat_workers] 파일={files}  박스={boxes}  스킵={skipped}")


def write_data_yaml():
    """YOLO 학습용 data.yaml 생성"""
    yaml_path = OUT_DIR / "data.yaml"
    content = (
        f"names:\n"
        + "".join(f"- {name}\n" for name in NEW_CLASS_NAMES)
        + f"nc: {len(NEW_CLASS_NAMES)}\n"
        f"train: {OUT_DIR / 'train' / 'images'}\n"
        f"val: {OUT_DIR / 'valid' / 'images'}\n"
        f"test: {OUT_DIR / 'test' / 'images'}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")
    print(f"\n[✓] data.yaml 생성: {yaml_path}")


def main():
    if OUT_DIR.exists():
        print(f"[!] 기존 {OUT_DIR} 삭제 후 재생성")
        shutil.rmtree(OUT_DIR)

    print("=" * 60)
    print("[1/3] merged 데이터셋에서 헬멧 라벨 추출")
    print("=" * 60)
    for split in ("train", "valid", "test"):
        process_merged_split(split)

    print("\n" + "=" * 60)
    print("[2/3] Hard Hat Workers 데이터셋 병합 (train 에만 추가)")
    print("=" * 60)
    process_check_helmets()

    print("\n" + "=" * 60)
    print("[3/3] data.yaml 생성")
    print("=" * 60)
    write_data_yaml()

    # 최종 통계
    print("\n" + "=" * 60)
    print("[최종 통계]")
    for split in ("train", "valid", "test"):
        lbl_dir = OUT_DIR / split / "labels"
        img_dir = OUT_DIR / split / "images"
        if lbl_dir.exists():
            nl = len(list(lbl_dir.glob("*.txt")))
            ni = sum(1 for _ in img_dir.glob("*") if _.is_file())
            print(f"  {split:6s}: 라벨={nl:5d}  이미지={ni:5d}")
    print("=" * 60)


if __name__ == "__main__":
    main()

# ============================================================
# 웹캠 헬멧 샘플 수집 스크립트
# - 스페이스바: 현재 프레임 저장
# - h / n: 저장 클래스 토글 (helmet / no_helmet)
# - q / ESC: 종료
#
# 저장 위치:
#   datasets/webcam_helmet/raw/helmet/helmet_YYYYMMDD_HHMMSS_xxx.jpg
#   datasets/webcam_helmet/raw/no_helmet/no_helmet_YYYYMMDD_HHMMSS_xxx.jpg
#
# 라벨링은 아직 하지 않음. 촬영 후 Roboflow 업로드 또는 LabelImg 사용.
# 한글 경로 대응: cv2.imencode + buf.tofile 로 저장.
# ============================================================

import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent / "datasets" / "webcam_helmet" / "raw"
CLASSES = ["helmet", "no_helmet"]

CAM_INDEX = 0
CAM_W, CAM_H = 1280, 720


def imwrite_unicode(path: Path, img):
    ok, buf = cv2.imencode(path.suffix, img)
    if not ok:
        return False
    buf.tofile(str(path))
    return True


def count_saved():
    out = {}
    for c in CLASSES:
        d = ROOT / c
        out[c] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return out


def main():
    for c in CLASSES:
        (ROOT / c).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
    if not cap.isOpened():
        print("[err] 카메라 열기 실패")
        sys.exit(1)

    current_cls = "helmet"
    flash_until = 0.0

    print("=== 헬멧 샘플 수집 ===")
    print("SPACE : 저장")
    print("h / n : 클래스 토글 (helmet / no_helmet)")
    print("q/ESC : 종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[err] 프레임 읽기 실패")
            break

        disp = frame.copy()
        counts = count_saved()

        # 상단 오버레이
        color = (0, 200, 0) if current_cls == "helmet" else (0, 100, 255)
        cv2.rectangle(disp, (0, 0), (CAM_W, 60), (0, 0, 0), -1)
        cv2.putText(disp, f"class: {current_cls}", (20, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(disp,
                    f"saved  helmet:{counts['helmet']}  no_helmet:{counts['no_helmet']}",
                    (380, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(disp, "SPACE=save  h/n=class  q=quit",
                    (20, CAM_H - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        # 저장 플래시
        if time.time() < flash_until:
            cv2.rectangle(disp, (0, 60), (CAM_W, CAM_H - 40), (0, 255, 0), 8)

        cv2.imshow("Helmet Sample Collector", disp)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        elif key == ord("h"):
            current_cls = "helmet"
        elif key == ord("n"):
            current_cls = "no_helmet"
        elif key == 32:  # space
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            uid = uuid.uuid4().hex[:6]
            fname = f"{current_cls}_{ts}_{uid}.jpg"
            fpath = ROOT / current_cls / fname
            if imwrite_unicode(fpath, frame):
                print(f"[save] {fpath.name}  ({current_cls})")
                flash_until = time.time() + 0.15
            else:
                print(f"[err] 저장 실패: {fpath}")

    cap.release()
    cv2.destroyAllWindows()

    final = count_saved()
    print("\n=== 완료 ===")
    for c in CLASSES:
        print(f"  {c}: {final[c]}장  ->  {ROOT / c}")


if __name__ == "__main__":
    main()

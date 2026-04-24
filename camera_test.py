# ============================================================
# 노트북 카메라 실시간 PPE 감지 테스트
# Flask AI 서버(app.py)가 실행 중이어야 동작함
# 실행: python camera_test.py
# 종료: 영상 창에서 'q' 키
# ============================================================

import cv2
import requests
import time
import numpy as np
from PIL import ImageFont, ImageDraw, Image

# Flask AI 서버 주소
SERVER_URL = "http://127.0.0.1:8001/detect"

# 추론 간격 (초) - 너무 짧으면 서버 부하 증가
DETECT_INTERVAL = 0.3  # 300ms마다 한 번 추론

# 감지 결과 표시 색상 (RGB, Pillow 기준)
COLOR_PASS = (0, 200, 0)     # 초록: 감지됨
COLOR_FAIL = (220, 0, 0)     # 빨강: 미감지
COLOR_WHITE = (255, 255, 255) # 흰색: 안내 텍스트

# 클래스별 바운딩 박스 색상 (BGR, OpenCV 기준)
# 착용(positive)은 초록 계열, 미착용(negative)은 빨강 계열
BOX_COLORS = {
    "helmet":    (0, 220, 0),     # 초록 - 헬멧 착용
    "no_helmet": (0, 0, 220),     # 빨강 - 헬멧 미착용
    "mask":      (255, 140, 0),   # 주황 - 마스크 착용
    "no_mask":   (0, 80, 220),    # 빨강 - 마스크 미착용
    "gloves":    (220, 0, 220),   # 보라 - 장갑 착용
    "no_gloves": (40, 40, 200),   # 빨강 - 장갑 미착용
}

# 클래스 이름 → 한글 매핑
LABEL_KR = {
    "helmet":    "헬멧",
    "no_helmet": "헬멧X",
    "mask":      "마스크",
    "no_mask":   "마스크X",
    "gloves":    "장갑",
    "no_gloves": "장갑X",
}

# Windows 맑은 고딕 폰트 로드 (한글 지원)
FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
FONT_NORMAL = ImageFont.truetype(FONT_PATH, 34)   # 항목 텍스트 크기
FONT_LARGE  = ImageFont.truetype(FONT_PATH, 88)   # PASS/FAIL 크기
FONT_SMALL  = ImageFont.truetype(FONT_PATH, 24)   # 바운딩박스 라벨 크기


def put_text_kr(frame: np.ndarray, text: str, pos: tuple,
                font: ImageFont.FreeTypeFont, color: tuple) -> np.ndarray:
    """
    OpenCV 프레임에 한글 텍스트를 Pillow로 렌더링하여 반환
    OpenCV는 한글을 직접 지원하지 않으므로 PIL을 경유해야 함
    """
    # OpenCV BGR → PIL RGB 변환
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    draw.text(pos, text, font=font, fill=color)
    # PIL RGB → OpenCV BGR 변환 후 반환
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def send_frame(frame: np.ndarray) -> dict | None:
    """
    OpenCV 프레임을 JPEG로 인코딩하여 AI 서버로 전송
    서버 응답 또는 오류 시 None 반환
    """
    # OpenCV BGR 프레임 → JPEG 바이트 인코딩 (quality 80)
    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    image_bytes = buffer.tobytes()

    try:
        resp = requests.post(
            SERVER_URL,
            data=image_bytes,
            headers={"Content-Type": "image/jpeg"},
            timeout=2.0  # 2초 안에 응답 없으면 무시
        )
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        # 서버 연결 실패 시 조용히 넘김 (화면은 계속 표시)
        pass
    return None


def draw_boxes(frame: np.ndarray, boxes: list) -> np.ndarray:
    """
    서버가 반환한 바운딩 박스 좌표를 프레임에 직접 그린다.
    클래스별로 다른 색상과 한글 레이블을 표시하여 감지 위치를 시각화
    """
    for box in boxes:
        cls_name = box["class"]
        conf = box["conf"]
        x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]

        # 클래스별 색상 (BGR)
        color_bgr = BOX_COLORS.get(cls_name, (200, 200, 200))
        # Pillow 한글 텍스트용 RGB 변환
        color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

        # 바운딩 박스 사각형
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)

        # 레이블 배경 박스 (텍스트 가독성)
        label_text = f"{LABEL_KR.get(cls_name, cls_name)} {conf:.2f}"
        # 레이블이 화면 위로 넘어가면 박스 안쪽에 표시
        text_y = y1 - 8 if y1 > 36 else y2 + 8
        frame = put_text_kr(frame, label_text, (x1, text_y), FONT_SMALL, color_rgb)

    return frame


def draw_results(frame: np.ndarray, result: dict) -> np.ndarray:
    """
    감지 결과를 프레임 위에 오버레이로 표시
    - 좌상단: 항목별 감지 여부 + confidence
    - 우상단: 전체 PASS/FAIL
    - 감지된 객체 위치: 바운딩 박스 + 한글 레이블
    """
    if not result:
        # 서버 응답 없을 때 안내 메시지 표시
        frame = put_text_kr(frame, "서버 연결 중...", (20, 20), FONT_SMALL, (0, 165, 255))
        return frame

    detected = result.get("detected", {})
    confidences = result.get("confidences", {})
    boxes = result.get("boxes", [])  # 바운딩 박스 좌표 목록

    # ① 바운딩 박스를 먼저 그림 (텍스트 오버레이 아래에 위치)
    frame = draw_boxes(frame, boxes)

    # 표시할 항목 순서 (소문자 키)
    items = [
        ("helmet", "헬멧"),
        ("mask",   "마스크"),
    ]

    # ② 좌상단 반투명 배경 박스
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 124), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # ③ 항목별 감지 상태 텍스트
    for i, (key, label) in enumerate(items):
        is_detected = detected.get(key, False)
        conf = confidences.get(key, 0.0)

        color = COLOR_PASS if is_detected else COLOR_FAIL
        status = f"O  {conf:.2f}" if is_detected else "X"
        text = f"{label}: {status}"

        y_pos = 20 + i * 46
        frame = put_text_kr(frame, text, (24, y_pos), FONT_NORMAL, color)

    # ④ 전체 PASS/FAIL 판정
    all_pass = all(detected.get(k, False) for k, _ in items)
    overall_text = "PASS" if all_pass else "FAIL"
    overall_color = COLOR_PASS if all_pass else COLOR_FAIL

    h, w = frame.shape[:2]
    frame = put_text_kr(frame, overall_text, (w - 220, 20), FONT_LARGE, overall_color)

    return frame


def main():
    # 기본 카메라(인덱스 0) 열기 — Windows는 DSHOW 백엔드가 더 안정적
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("[오류] 카메라를 열 수 없습니다.")
        return

    # 해상도 설정 (HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 창 이름은 ASCII만 사용 (OpenCV 타이틀바 한글 미지원)
    WIN = "PPE Detection (q: quit)"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)

    print("[안내] 카메라 실행 중... 'q' 키로 종료")

    last_detect_time = 0.0
    last_result = None  # 마지막 추론 결과 (프레임마다 재사용)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[오류] 프레임을 읽을 수 없습니다.")
            break

        # 좌우 반전(거울 모드) — 사용자 시점 맞추기
        frame = cv2.flip(frame, 1)

        now = time.time()

        # 설정한 간격마다 서버에 추론 요청
        if now - last_detect_time >= DETECT_INTERVAL:
            last_result = send_frame(frame)
            last_detect_time = now

        # 마지막 결과를 프레임에 오버레이
        frame = draw_results(frame, last_result)

        # 창에 출력
        cv2.imshow(WIN, frame)

        # 'q' 키 입력 시 종료
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[종료] 카메라 테스트 종료")


if __name__ == "__main__":
    main()

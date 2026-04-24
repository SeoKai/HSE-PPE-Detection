# ============================================================
# Flask AI 서버 - PPE 안전장비 감지 추론 API
# 3개 모듈을 통합한 하이브리드 파이프라인
#   - YOLO(helmet) + MediaPipe+MobileNet(mask) + MediaPipe+MobileNet(gloves)
# 포트: 8001
# ============================================================

from flask import Flask, request, jsonify
from model.detector import PPEDetector
import os

app = Flask(__name__)

# -----------------------------------------------
# 서버 시작 시 3개 가중치를 한 번만 로드
# -----------------------------------------------
BASE = os.path.dirname(__file__)
HELMET_W = os.path.join(BASE, "weights", "helmet_best.pt")
MASK_W   = os.path.join(BASE, "weights", "mask_cls.pt")
GLOVES_W = os.path.join(BASE, "weights", "gloves_cls.pt")

missing = [p for p in (HELMET_W, MASK_W, GLOVES_W) if not os.path.exists(p)]
if missing:
    print("[경고] 다음 가중치가 없습니다. 해당 항목은 스킵됩니다:")
    for p in missing:
        print(f"   - {p}")

detector = PPEDetector(
    helmet_weight=HELMET_W,
    mask_weight=MASK_W,
    gloves_weight=GLOVES_W,
)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "modules": {
            "helmet": detector.helmet is not None,
            "mask":   detector.mask   is not None,
            "gloves": detector.gloves is not None,
        },
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    단일 프레임 추론 엔드포인트

    요청 (두 가지 지원):
      1) multipart/form-data: key='file', JPEG
      2) raw binary: Content-Type: image/jpeg

    응답 예시:
    {
        "detected":    {"helmet": true, "mask": false, "gloves": true},
        "confidences": {"helmet": 0.93, "mask": 0.81, "gloves": 0.74},
        "boxes": [
            {"class": "helmet",    "conf": 0.93, "x1":..., "y1":..., "x2":..., "y2":...},
            {"class": "no_mask",   "conf": 0.81, ...},
            {"class": "gloves",    "conf": 0.74, ...}
        ]
    }
    """
    if request.files.get("file"):
        image_bytes = request.files["file"].read()
    elif request.data:
        image_bytes = request.data
    else:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    if len(image_bytes) == 0:
        return jsonify({"error": "빈 이미지 데이터입니다."}), 400

    result = detector.infer(image_bytes)
    if "error" in result:
        return jsonify(result), 422

    return jsonify(result)


if __name__ == "__main__":
    # threaded=True: 여러 프레임 동시 처리
    # host="0.0.0.0": 같은 네트워크의 백엔드에서 접근 가능
    app.run(host="0.0.0.0", port=8001, threaded=True, debug=False)
# ============================================================
# Flask AI 서버 - PPE 안전장비 감지 추론 API
# 백엔드 서버가 이 서버로 프레임을 전달하면 YOLOv8로 추론 후 결과 반환
# 포트: 8001 (백엔드와 분리 운영)
# ============================================================

from flask import Flask, request, jsonify
from model.detector import PPEDetector
import os

app = Flask(__name__)

# -----------------------------------------------
# 서버 시작 시 모델 1회만 로드
# 매 요청마다 로드하면 ~300ms 지연 발생 → 반드시 전역 로드
# -----------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "ppe_6cls_best.pt")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"모델 가중치 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
        "train.py를 먼저 실행하여 학습을 완료하세요."
    )

detector = PPEDetector(MODEL_PATH)


@app.route("/health", methods=["GET"])
def health():
    """
    백엔드 서버가 AI 서버 상태를 확인할 때 호출
    응답: {"status": "ok", "model": "ppe_best.pt"}
    """
    return jsonify({
        "status": "ok",
        "model": os.path.basename(MODEL_PATH)
    })


@app.route("/detect", methods=["POST"])
def detect():
    """
    단일 프레임 추론 엔드포인트
    ---
    요청 방식 (두 가지 모두 지원):
      1. multipart/form-data: key='file', value=JPEG 이미지
      2. raw binary: Content-Type: image/jpeg

    응답 예시:
    {
        "detected": {
            "Gloves": true,
            "Hardhat": true,
            "Mask": false
        },
        "confidences": {
            "Gloves": 0.872,
            "Hardhat": 0.934,
            "Mask": 0.0
        }
    }
    """
    # 이미지 바이트 추출 (두 가지 요청 방식 지원)
    if request.files.get("file"):
        # multipart/form-data 방식
        image_bytes = request.files["file"].read()
    elif request.data:
        # raw binary 방식 (WebSocket 프레임 전달에 적합)
        image_bytes = request.data
    else:
        return jsonify({"error": "이미지 데이터가 없습니다."}), 400

    # 빈 데이터 체크
    if len(image_bytes) == 0:
        return jsonify({"error": "빈 이미지 데이터입니다."}), 400

    # 추론 실행
    result = detector.infer(image_bytes)

    # 추론 내부 오류 발생 시 (잘못된 이미지 포맷 등)
    if "error" in result:
        return jsonify(result), 422

    return jsonify(result)


if __name__ == "__main__":
    # threaded=True: 여러 프레임을 동시에 처리하기 위한 멀티스레드 모드
    # host="0.0.0.0": 같은 네트워크의 백엔드 서버에서 접근 가능
    app.run(host="0.0.0.0", port=8001, threaded=True, debug=False)

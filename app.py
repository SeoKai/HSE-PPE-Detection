# ============================================================
# Flask AI 서버 - PPE 안전장비 감지 추론 API
# 3개 모듈을 통합한 하이브리드 파이프라인
#   - YOLO(helmet) + MediaPipe+MobileNet(mask) + MediaPipe+MobileNet(gloves)
# 포트: 8001
# ============================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
from model.detector import PPEDetector
import os

app = Flask(__name__)

# BN 백엔드에서 서버-서버로 호출하므로 localhost만 허용
# 로컬 개발 시 브라우저 직접 호출이 필요하면 "*" 로 변경
CORS(app, resources={r"/*": {"origins": ["http://localhost:8080", "http://localhost:3001", "http://211.188.55.206:3001"]}})
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

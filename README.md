# HSE PPE Detection

산업 현장 작업자의 **개인 보호구(PPE, Personal Protective Equipment)** 착용 여부를 웹캠 영상에서 실시간으로 판별하는 시스템입니다. Flask AI 서버가 클라이언트(카메라)로부터 JPEG 프레임을 받아 **헬멧 / 마스크** 착용 여부를 분류하여 반환합니다.

---

## 🎯 최종 결과

| 항목   | 모델                          | 지표    | 성능             |
| ------ | ----------------------------- | ------- | ---------------- |
| 헬멧   | YOLOv8n (fine-tuned)          | mAP50   | **0.995**        |
| 마스크 | MediaPipe Face + MobileNetV3  | Val Acc | **0.975**        |
| 장갑   | MediaPipe Hands + MobileNetV3 | Val Acc | 0.945 _(내부용)_ |

> 장갑 분류기는 실제 웹캠 환경에서 신뢰도가 낮아 **최종 출력에서는 제외**했지만, 손 박스 좌표는 헬멧/마스크 모듈의 오탐 필터로 계속 활용합니다.

---

## 🏗️ 아키텍처

```
[Webcam] ──JPEG──▶ [Flask /detect] ──▶ [PPEDetector]
                                          │
                                          ├─ 1) GlovesClassifier  → hand_boxes 추출 (필터용)
                                          ├─ 2) HelmetDetector    → YOLO 검출 + hand IoU 필터
                                          └─ 3) MaskClassifier    → Face 검출 + hand IoU 필터 + MobileNetV3
                                          │
                                          ▼
                                      {detected, confidences, boxes} (JSON)
```

3개 서브 모듈은 **순서가 중요**합니다. 장갑 모듈이 먼저 돌아 손 좌표를 수집한 뒤, 헬멧과 마스크 모듈이 그 좌표를 받아 **손바닥을 얼굴·헬멧으로 오인하는 경우를 차단**합니다.

---

## 각 모듈 로직

### 1. 헬멧 (YOLOv8)

**문제**: 공개 데이터셋(Roboflow PPE 17K장)은 원거리 건설현장 사진 위주로 학습되어, 웹캠 클로즈업 환경에서 **손을 헬멧으로 오탐**하고 정작 머리 위 헬멧을 놓치는 도메인 미스매치가 심각했습니다.

**해결**:

1. **데이터 품질 필터링** — `scripts/tools/filter_dataset.py`로 17K 장 중 깨진 이미지·너무 작은 이미지·헬멧 bbox 면적 0.1% 미만 샘플 **1,494장 제거**.
2. **근거리 웹캠 데이터 자체 구축** — `scripts/tools/collect_helmet_samples.py`로 직접 촬영 → Roboflow 라벨링 183장 → 80/15/5 split.
3. **Fine-tune** — `scripts/train/finetune_helmet_webcam.py`
   - `lr0=5e-5`, `epochs=25`, `imgsz=640`, AdamW, `weight_decay=5e-4`
   - 증강 최소화: `mosaic=0`, `mixup=0`, `copy_paste=0`, `degrees=5`, `scale=0.2`
4. **기하 필터 → Hand IoU 필터로 교체** — fine-tune 후엔 기하(종횡비, 중심y, 최소면적) 필터를 꺼도 오탐이 거의 없어, 손 박스와 IoU ≥ 0.35 겹치는 검출만 거부하는 방식으로 단순화.

### 2. 마스크 (MediaPipe + MobileNetV3)

**2-stage 파이프라인**:

1. **MediaPipe Face Detector**로 얼굴 영역을 먼저 찾고,
2. **MobileNetV3-small** 분류기로 해당 crop이 `mask` / `no_mask`인지 판단.

**개선 포인트**:

- **Short-range → Full-range 모델 교체** (`blaze_face_full_range.tflite`) — 원거리 얼굴 검출율 개선.
- **`face_min_conf = 0.5`** — 손바닥/팔을 얼굴로 오탐하는 저신뢰 검출을 필터링.
- **Hand IoU 필터** — 감지된 얼굴 박스가 MediaPipe Hands 결과와 IoU ≥ 0.35이면 손바닥 오탐으로 간주하고 스킵.

### 3. 장갑 (MediaPipe Hands + MobileNetV3)

MediaPipe HandLandmarker로 손 영역을 얻고, MobileNetV3-small로 `gloves` / `no_gloves` 분류. 웹캠 환경 일반화가 어려워 **출력 JSON에서 장갑 항목은 노출하지 않고**, 손 좌표만 다른 모듈의 오탐 필터로 재활용합니다.

---

## 프로젝트 구조

```
hse_ai/
├── app.py                     # Flask 서버 (POST /detect)
├── camera_test.py             # 로컬 웹캠 클라이언트 (300ms 주기)
├── model/
│   ├── detector.py            # PPEDetector: 3개 모듈 오케스트레이션
│   ├── helmet_detector.py     # YOLOv8 + hand IoU 필터
│   ├── mask_classifier.py     # MediaPipe Face + MobileNetV3 + hand IoU 필터
│   ├── gloves_classifier.py   # MediaPipe Hands + MobileNetV3
│   └── classifier_utils.py
├── weights/                   # 가중치 (git ignored)
│   ├── helmet_best.pt
│   ├── mask_cls.pt
│   ├── gloves_cls.pt
│   └── mp/                    # MediaPipe tflite 파일
├── datasets/                  # 데이터셋 (git ignored)
└── scripts/
    ├── train/                 # fine-tune, train 스크립트
    ├── prepare/               # 데이터셋 준비/크롭
    └── tools/                 # filter_dataset, collect_helmet_samples
```

---

## 실행 방법

### 1. 환경 설정

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -r requirements.txt
```

### 2. 가중치 준비

`weights/` 하위에 다음 파일을 배치:

- `helmet_best.pt` — YOLOv8 헬멧 검출 fine-tuned
- `mask_cls.pt` — 마스크 MobileNetV3 분류기
- `gloves_cls.pt` — 장갑 MobileNetV3 분류기
- `mp/blaze_face_full_range.tflite` — MediaPipe Face (full-range)
- `mp/hand_landmarker.task` — MediaPipe Hands

### 3. 서버 실행

```bash
python app.py
# → Running on http://127.0.0.1:8001
```

### 4. 실시간 테스트

```bash
python camera_test.py
```

---

## API

### `POST /detect`

**Request**: JPEG 이미지 바이너리 (`Content-Type: image/jpeg`)

**Response**:

```json
{
  "detected": { "helmet": true, "mask": false },
  "confidences": { "helmet": 0.93, "mask": 0.81 },
  "boxes": [
    {
      "class": "helmet",
      "conf": 0.93,
      "x1": 120,
      "y1": 45,
      "x2": 280,
      "y2": 180
    },
    {
      "class": "no_mask",
      "conf": 0.81,
      "x1": 150,
      "y1": 200,
      "x2": 260,
      "y2": 310
    }
  ]
}
```

### `GET /health`

서버 상태 및 로드된 모듈 확인.

---

## 주요 개선 사항 요약

| #   | 문제                                                 | 해결                                                  |
| --- | ---------------------------------------------------- | ----------------------------------------------------- |
| 1   | 공개 헬멧 데이터셋 도메인 미스매치 (원거리 건설현장) | 웹캠 데이터 직접 수집·라벨링 후 fine-tune             |
| 2   | 17K 중 품질 불량 샘플 다수                           | 자동 필터로 1,494장 제거                              |
| 3   | 손바닥 → no_helmet / no_mask 오탐                    | MediaPipe Hands로 얻은 손 좌표와 IoU ≥ 0.35 검출 거부 |
| 4   | 원거리 마스크 미검출                                 | MediaPipe Face **full-range** 모델로 교체             |
| 5   | 장갑 분류기 실환경 일반화 실패                       | 출력에서 제외, 손 좌표만 필터용으로 재사용            |
| 6   | Windows + Python 3.14 DataLoader 크래시              | `workers=0`, `cache="ram"` 고정                       |

---

## 학습 재현

```bash
# 웹캠 데이터 수집
python scripts/tools/collect_helmet_samples.py

# 데이터 품질 필터링 (merged 데이터셋)
python scripts/tools/filter_dataset.py --apply

# 데이터셋 준비
python scripts/prepare/prepare_webcam_helmet.py

# Fine-tune
python scripts/train/finetune_helmet_webcam.py
```

---

## Stack

- **Python** 3.14, **PyTorch** 2.11 (CUDA 12.6)
- **Ultralytics YOLOv8** 8.4
- **MediaPipe Tasks** (FaceDetector, HandLandmarker)
- **torchvision** (MobileNetV3-small)
- **Flask** (추론 서버), **OpenCV**, **Pillow**

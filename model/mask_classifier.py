# ============================================================
# 마스크 분류 파이프라인 (MediaPipe Tasks API)
# - MediaPipe FaceDetector (Tasks API, tflite) 로 얼굴 ROI 추출
# - MobileNetV3-Small 분류기로 mask/no_mask 판정
# ============================================================

import os
import cv2
import torch
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from .classifier_utils import load_classifier

_MP_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "mp",
)
# full-range 우선 (원거리 대응). 없으면 기본 short-range로 fallback.
_MP_MODEL = os.path.join(_MP_DIR, "blaze_face_full_range.tflite")
if not os.path.exists(_MP_MODEL):
    _MP_MODEL = os.path.join(_MP_DIR, "face_detector.tflite")


class MaskClassifier:
    def __init__(self, weight_path: str,
                 conf_threshold: float = 0.7,
                 face_min_conf: float = 0.5,
                 pad_ratio: float = 0.10,
                 hand_iou_reject: float = 0.35):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names, self.img_size, self.tf = load_classifier(
            weight_path, self.device
        )
        self.conf_threshold = conf_threshold
        self.pad_ratio = pad_ratio
        self.hand_iou_reject = hand_iou_reject

        with open(_MP_MODEL, "rb") as f:
            model_bytes = f.read()
        opts = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            min_detection_confidence=face_min_conf,
        )
        self.mp_face = mp_vision.FaceDetector.create_from_options(opts)

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / (area_a + area_b - inter)

    def _pad_square(self, x1, y1, x2, y2, W, H):
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        side = max(x2 - x1, y2 - y1) * (1 + self.pad_ratio * 2)
        half = side / 2
        nx1 = max(0, int(round(cx - half)))
        ny1 = max(0, int(round(cy - half)))
        nx2 = min(W, int(round(cx + half)))
        ny2 = min(H, int(round(cy + half)))
        return nx1, ny1, nx2, ny2

    def classify(self, img_bgr, hand_boxes=None) -> list:
        H, W = img_bgr.shape[:2]
        hand_boxes = hand_boxes or []
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.mp_face.detect(mp_img)

        out = []
        if not result.detections:
            return out

        crops_tensors = []
        boxes_pixel = []
        for det in result.detections:
            bbox = det.bounding_box  # origin_x, origin_y, width, height (pixels)
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = x1 + int(bbox.width)
            y2 = y1 + int(bbox.height)
            x1, y1, x2, y2 = self._pad_square(x1, y1, x2, y2, W, H)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue
            # 손과 겹치는 얼굴 박스는 오탐 (손바닥)으로 간주해 거부
            if any(self._iou((x1, y1, x2, y2), hb) >= self.hand_iou_reject for hb in hand_boxes):
                continue
            crop_bgr = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(crop_rgb)
            tensor = self.tf(pil)
            crops_tensors.append(tensor)
            boxes_pixel.append((x1, y1, x2, y2))

        if not crops_tensors:
            return out

        batch = torch.stack(crops_tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for (x1, y1, x2, y2), p in zip(boxes_pixel, probs):
            pred_idx = int(np.argmax(p))
            conf = float(p[pred_idx])
            if conf < self.conf_threshold:
                continue
            out.append({
                "class": self.class_names[pred_idx],
                "conf": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        return out

    def close(self):
        try:
            self.mp_face.close()
        except Exception:
            pass

# ============================================================
# 장갑 분류 파이프라인 (MediaPipe Tasks API)
# - MediaPipe HandLandmarker(.task) 로 손 랜드마크 → 바운딩 박스
# - MobileNetV3-Small 분류기로 gloves/no_gloves 판정
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

_MP_MODEL = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "weights", "mp", "hand_landmarker.task",
)


class GlovesClassifier:
    def __init__(self, weight_path: str,
                 conf_threshold: float = 0.85,
                 hand_min_conf: float = 0.5,
                 pad_ratio: float = 0.15,
                 max_hands: int = 4,
                 min_hand_area_ratio: float = 0.008,
                 min_prob_margin: float = 0.55,
                 max_skin_ratio: float = 0.25):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.class_names, self.img_size, self.tf = load_classifier(
            weight_path, self.device
        )
        self.conf_threshold = conf_threshold
        self.pad_ratio = pad_ratio
        self.min_hand_area_ratio = min_hand_area_ratio  # 이미지 대비 손 bbox 최소 면적 비율
        self.min_prob_margin = min_prob_margin          # gloves/no_gloves 확률 차이 최소값
        self.max_skin_ratio = max_skin_ratio            # gloves 양성시 피부색 픽셀 비율 상한(초과시 맨손으로 판정)

        with open(_MP_MODEL, "rb") as f:
            model_bytes = f.read()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
            num_hands=max_hands,
            min_hand_detection_confidence=hand_min_conf,
            min_hand_presence_confidence=hand_min_conf,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE,
        )
        self.mp_hands = mp_vision.HandLandmarker.create_from_options(opts)

    def _landmarks_to_box(self, landmarks, W, H):
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        x1 = int(round(min(xs) * W))
        y1 = int(round(min(ys) * H))
        x2 = int(round(max(xs) * W))
        y2 = int(round(max(ys) * H))
        return x1, y1, x2, y2

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

    @staticmethod
    def _skin_ratio(crop_bgr) -> float:
        """HSV + YCrCb 기반 피부색 픽셀 비율 (맨손은 높음, 장갑은 낮음)"""
        if crop_bgr.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        # HSV 피부색 범위 (동양인/서양인 공통 대략)
        mask_hsv = cv2.inRange(hsv, (0, 30, 60), (25, 180, 255))
        # YCrCb 피부색 범위 (Cr: 133~173, Cb: 77~127)
        mask_ycb = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
        mask = cv2.bitwise_and(mask_hsv, mask_ycb)
        return float(mask.sum()) / (255.0 * mask.shape[0] * mask.shape[1])

    def classify(self, img_bgr) -> list:
        H, W = img_bgr.shape[:2]
        img_area = float(H * W)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = self.mp_hands.detect(mp_img)

        out = []
        if not result.hand_landmarks:
            return out

        # 클래스 인덱스 식별 (gloves / no_gloves)
        try:
            gloves_idx = self.class_names.index("gloves")
        except ValueError:
            gloves_idx = 0

        crops_tensors = []
        boxes_pixel = []
        skin_ratios = []
        for landmarks in result.hand_landmarks:
            x1, y1, x2, y2 = self._landmarks_to_box(landmarks, W, H)
            x1, y1, x2, y2 = self._pad_square(x1, y1, x2, y2, W, H)
            bw = x2 - x1
            bh = y2 - y1
            if bw < 16 or bh < 16:
                continue
            # 손 크기 필터: 너무 작으면(멀리 있는 손) 판정 포기
            if (bw * bh) / img_area < self.min_hand_area_ratio:
                continue
            crop_bgr = img_bgr[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(crop_rgb)
            tensor = self.tf(pil)
            crops_tensors.append(tensor)
            boxes_pixel.append((x1, y1, x2, y2))
            skin_ratios.append(self._skin_ratio(crop_bgr))

        if not crops_tensors:
            return out

        batch = torch.stack(crops_tensors, dim=0).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        for (x1, y1, x2, y2), p, skin in zip(boxes_pixel, probs, skin_ratios):
            pred_idx = int(np.argmax(p))
            conf = float(p[pred_idx])
            if conf < self.conf_threshold:
                continue
            # gloves 양성 판정: margin 게이팅 + 피부색 비율 게이팅 (맨손 오탐 차단)
            if pred_idx == gloves_idx:
                other = float(1.0 - p[gloves_idx])
                margin = float(p[gloves_idx]) - other
                if margin < self.min_prob_margin:
                    continue
                if skin > self.max_skin_ratio:
                    # 피부색 비율 너무 높음 → 맨손으로 재분류
                    try:
                        no_gloves_idx = self.class_names.index("no_gloves")
                    except ValueError:
                        continue
                    out.append({
                        "class": "no_gloves",
                        "conf": round(float(p[no_gloves_idx]), 3),
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    })
                    continue
            out.append({
                "class": self.class_names[pred_idx],
                "conf": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        return out

    def close(self):
        try:
            self.mp_hands.close()
        except Exception:
            pass

# ============================================================
# PPE 통합 오케스트레이터
# - Helmet: YOLOv8 (weights/helmet_best.pt)
# - Mask  : MediaPipe FaceDetection + MobileNetV3 (weights/mask_cls.pt)
# - Gloves: MediaPipe Hands + MobileNetV3 (weights/gloves_cls.pt)
#
# 출력 스키마:
# {
#   "detected":    {"helmet": bool, "mask": bool, "gloves": bool},
#   "confidences": {"helmet": float, "mask": float, "gloves": float},
#   "boxes": [
#       {"class": "helmet"|"no_helmet"|"mask"|"no_mask"|"gloves"|"no_gloves",
#        "conf": float, "x1,y1,x2,y2": int},
#       ...
#   ]
# }
# ============================================================

import os
import cv2
import numpy as np

from .helmet_detector import HelmetDetector
from .mask_classifier import MaskClassifier
from .gloves_classifier import GlovesClassifier


class PPEDetector:
    """
    3개 서브 모듈을 통합하여 헬멧/마스크/장갑 착용 여부를 판단.
    각 서브 모듈은 선택적으로 로드 (가중치 파일 없으면 해당 항목 스킵).
    """

    ITEM_KEYS = ("helmet", "mask", "gloves")

    def __init__(self,
                 helmet_weight: str,
                 mask_weight: str,
                 gloves_weight: str,
                 helmet_conf: float = 0.50,
                 mask_conf: float = 0.7,
                 gloves_conf: float = 0.85):
        # 도메인-추가학습된 helmet 모델 기준: 기하 필터는 비활성,
        # 손과 겹치는 box만 오탐으로 거부.
        self.helmet = HelmetDetector(helmet_weight, conf_threshold=helmet_conf,
                                     max_center_y_ratio=1.0,
                                     aspect_min=0.0,
                                     aspect_max=999.0,
                                     min_area_ratio=0.0,
                                     hand_iou_reject=0.35) \
            if os.path.exists(helmet_weight) else None
        self.mask = MaskClassifier(mask_weight, conf_threshold=mask_conf) \
            if os.path.exists(mask_weight) else None
        self.gloves = GlovesClassifier(gloves_weight, conf_threshold=gloves_conf) \
            if os.path.exists(gloves_weight) else None

        loaded = []
        if self.helmet: loaded.append("helmet")
        if self.mask:   loaded.append("mask")
        if self.gloves: loaded.append("gloves")
        print(f"[PPEDetector] loaded modules: {loaded}")

    # ----------------------------------------------------
    # positive(착용) 박스가 있으면 detected=True + 해당 conf,
    # 없고 negative(미착용)만 있으면 detected=False + 해당 conf
    # ----------------------------------------------------
    @staticmethod
    def _aggregate(boxes: list, positive_cls: str, negative_cls: str):
        best_pos = 0.0
        best_neg = 0.0
        for b in boxes:
            if b["class"] == positive_cls and b["conf"] > best_pos:
                best_pos = b["conf"]
            elif b["class"] == negative_cls and b["conf"] > best_neg:
                best_neg = b["conf"]
        if best_pos > 0:
            return True, best_pos
        return False, best_neg

    def infer(self, image_bytes: bytes) -> dict:
        """JPEG 바이트 → 통합 결과"""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "이미지를 디코딩할 수 없습니다. JPEG 포맷인지 확인하세요."}
        return self.infer_image(img)

    def infer_image(self, img_bgr) -> dict:
        """OpenCV BGR 이미지 → 통합 결과"""
        all_boxes = []

        # 1) 장갑 모듈은 손 박스 수집용으로만 실행 (mask/helmet 오탐 필터에 사용).
        #    장갑 분류 결과 자체는 현재 웹캠 환경에서 신뢰도가 낮아 출력에서 제외.
        hand_boxes = []
        if self.gloves is not None:
            glove_boxes = self.gloves.classify(img_bgr)
            for b in glove_boxes:
                hand_boxes.append((b["x1"], b["y1"], b["x2"], b["y2"]))

        # 2) 헬멧: 손 박스와 겹치는 것은 오탐으로 거부
        if self.helmet is not None:
            all_boxes.extend(self.helmet.detect(img_bgr, hand_boxes=hand_boxes))

        # 3) 마스크 (손과 겹치는 얼굴 박스는 손바닥 오탐으로 거부)
        if self.mask is not None:
            all_boxes.extend(self.mask.classify(img_bgr, hand_boxes=hand_boxes))

        detected = {}
        confidences = {}
        for key, pos, neg in (
            ("helmet", "helmet", "no_helmet"),
            ("mask",   "mask",   "no_mask"),
        ):
            is_det, conf = self._aggregate(all_boxes, pos, neg)
            detected[key] = is_det
            confidences[key] = round(conf, 3)

        return {
            "detected": detected,
            "confidences": confidences,
            "boxes": all_boxes,
        }

    def close(self):
        if self.mask:   self.mask.close()
        if self.gloves: self.gloves.close()

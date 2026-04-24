# ============================================================
# 헬멧 전용 YOLO 감지기
# - 2클래스: helmet / no_helmet
# - 후처리 필터 (helmet 양성 박스에만 적용):
#   * 세로 위치: 박스 중심 y가 이미지 상단 60% 이내
#   * 종횡비(w/h): 0.6 ~ 1.8
#   * 최소 크기: 이미지의 0.3% 이상
# ============================================================

from ultralytics import YOLO


class HelmetDetector:
    CLASS_NAMES = ("helmet", "no_helmet")

    def __init__(self, model_path: str,
                 conf_threshold: float = 0.50,
                 max_center_y_ratio: float = 0.55,
                 aspect_min: float = 0.7,
                 aspect_max: float = 1.8,
                 min_area_ratio: float = 0.003,
                 hand_iou_reject: float = 0.35):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.max_center_y_ratio = max_center_y_ratio
        self.aspect_min = aspect_min
        self.aspect_max = aspect_max
        self.min_area_ratio = min_area_ratio
        self.hand_iou_reject = hand_iou_reject  # 손 박스와 IoU 이 값 이상이면 헬멧 오탐으로 간주
        self.class_map = self.model.names  # {0: 'helmet', 1: 'no_helmet'}

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

    def detect(self, img_bgr, hand_boxes=None) -> list:
        H, W = img_bgr.shape[:2]
        img_area = float(H * W)
        hand_boxes = hand_boxes or []

        results = self.model(img_bgr, conf=self.conf_threshold, verbose=False)[0]
        out = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = self.class_map.get(cls_id, str(cls_id))
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)
            area = bw * bh
            cy = (y1 + y2) / 2.0
            aspect = bw / bh

            if cls_name == "helmet":
                if cy / H > self.max_center_y_ratio:
                    continue
                if aspect < self.aspect_min or aspect > self.aspect_max:
                    continue
                if area / img_area < self.min_area_ratio:
                    continue

            # helmet/no_helmet 둘 다: 손 박스와 겹치면 오탐으로 거부
            if cls_name in ("helmet", "no_helmet"):
                rejected = False
                for hb in hand_boxes:
                    if self._iou((x1, y1, x2, y2), hb) >= self.hand_iou_reject:
                        rejected = True
                        break
                if rejected:
                    continue

            out.append({
                "class": cls_name,
                "conf": round(conf, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            })
        return out

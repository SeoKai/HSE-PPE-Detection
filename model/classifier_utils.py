# ============================================================
# 분류기 공통 로더 (MobileNetV3-Small)
# weights/*.pt 파일의 state_dict + class_names 를 읽어 모델 구성
# ============================================================

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small
from torchvision import transforms


def load_classifier(weight_path: str, device: torch.device):
    """
    weight_path: torch.save({'state_dict','class_names','arch','img_size'}) 로 저장된 파일
    반환: (model, class_names, img_size, transform)
    """
    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    class_names = ckpt["class_names"]
    img_size = ckpt.get("img_size", 224)
    arch = ckpt.get("arch", "mobilenet_v3_small")

    if arch != "mobilenet_v3_small":
        raise ValueError(f"지원하지 않는 아키텍처: {arch}")

    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    # inference 전처리 (학습 val 과 동일)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return model, class_names, img_size, tf

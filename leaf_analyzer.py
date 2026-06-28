"""
leaf_analyzer.py

"""

import os
import json
import cv2
import numpy as np
import requests
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

import torch
from torchvision import transforms


# =====================================================
# PATH
# =====================================================

YOLO_MODEL = "FullModel/modelyolo/yolo_leaf.pt"

CNN_MODEL = (
    "FullModel/modelcnn/"
    "mobilenet.pt"
)

IMAGE_DIR = (
    "FullDataset/camera"
)

FEEDBACK_DIR = (
    "FullDataset/leaf_feedback"
)

LAST_IRRIGATION = (
    f"{FEEDBACK_DIR}/"
    "last_irrigation.txt"
)

RESULT_FILE = (
    f"{FEEDBACK_DIR}/"
    "leaf_feedback.json"
)

CLASS_NAMES = [
    "dry",
    "healthy",
    "wet"
]

IMG_SIZE = 224

# =====================================================
# CAMERA
# =====================================================

ESP32_IP = "192.168.1.201"
ESP32_PORT = 80

ESP32_CAM_URL = (
    f"http://{ESP32_IP}:{ESP32_PORT}/capture"
)

PHOTO_HOURS = [
    6, 7, 8, 9, 10,
    15, 16, 17, 18, 19
]

_last_capture = None
# =====================================================
# LOAD MODELS
# =====================================================

yolo_model = None
cnn_model = None


def load_models():

    global yolo_model
    global cnn_model

    if yolo_model is None:

        yolo_model = YOLO(
            YOLO_MODEL
        )

        print("✅ YOLO loaded")

    if cnn_model is None:

        cnn_model = torch.load(
            CNN_MODEL,
            map_location="cpu",
            weights_only=False
        )

        cnn_model.eval()

        print("✅ CNN loaded")
# =====================================================
# CHỤP ẢNH ESP32
# =====================================================

def capture_image():

    try:

        response = requests.get(
            ESP32_CAM_URL,
            timeout=10
        )

        if response.status_code != 200:
            print("Không lấy được ảnh")
            return False

        Path(
            IMAGE_DIR
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        filename = datetime.now().strftime(
            "%Y%m%d_%H%M%S.jpg"
        )

        filepath = os.path.join(
            IMAGE_DIR,
            filename
        )

        with open(
            filepath,
            "wb"
        ) as f:

            f.write(
                response.content
            )

        print(
            f"📷 Đã lưu: {filename}"
        )

        return True

    except Exception as e:

        print(
            f"Lỗi ESP32-CAM: {e}"
        )

        return False
# =====================================================
# CHỤP THEO GIỜ
# =====================================================

def capture_if_needed():

    global _last_capture

    now = datetime.now()

    if now.hour not in PHOTO_HOURS:
        return

    if now.minute > 5:
        return

    stamp = now.strftime(
        "%Y%m%d_%H"
    )

    if stamp == _last_capture:
        return

    if capture_image():

        _last_capture = stamp
# =====================================================
# CNN
# =====================================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

def classify_leaf(img):

    img = transform(img)

    img = img.unsqueeze(0)

    with torch.no_grad():

        pred = cnn_model(img)

        pred = torch.softmax(pred,1)

    idx = pred.argmax(1).item()

    conf = pred[0][idx].item()

    label = CLASS_NAMES[idx]

    return label, conf


# =====================================================
# LẤY ẢNH SAU LẦN TƯỚI
# =====================================================

def get_images():

    if not os.path.exists(LAST_IRRIGATION):

        return sorted([
            os.path.join(IMAGE_DIR, f)
            for f in os.listdir(IMAGE_DIR)
            if f.lower().endswith(".jpg")
        ])

    with open(
        LAST_IRRIGATION,
        "r",
        encoding="utf8"
    ) as f:

        last_time = datetime.strptime(
            f.read().strip(),
            "%Y-%m-%d %H:%M:%S"
        )

    images = []

    for file in os.listdir(
        IMAGE_DIR
    ):

        if not file.lower().endswith(
            ".jpg"
        ):
            continue

        try:

            t = datetime.strptime(
                file.replace(
                    ".jpg",
                    ""
                ),
                "%Y%m%d_%H%M%S"
            )

            if t > last_time:

                images.append(
                    os.path.join(
                        IMAGE_DIR,
                        file
                    )
                )

        except:
            pass

    images.sort()

    return images


# =====================================================
# PHÂN TÍCH 1 ẢNH
# =====================================================

def analyze_image(image_path):

    image = cv2.imread(
        image_path
    )

    if image is None:

        return 0, 0, 0

    results = yolo_model.predict(
        image_path,
        conf=0.4,
        verbose=False
    )

    healthy = 0
    dry = 0
    wet = 0

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            crop = image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                continue

            label, conf = classify_leaf(
                crop
            )

            if label == "healthy":
                healthy += 1

            elif label == "dry":
                dry += 1

            elif label == "wet":
                wet += 1

    return healthy, dry, wet

# =====================================================
# ĐÁNH GIÁ GIỮA 2 LẦN TƯỚI
# =====================================================

def evaluate_leaf():

    load_models()

    images = get_images()
    if len(images)==0:

        print("Không có ảnh mới.")

        return {

            "time":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            "images":0,

            "healthy":0,

            "dry":0,

            "wet":0,

            "adjust_percent":0
        }
    total_h = 0
    total_d = 0
    total_w = 0

    print()
    print("===== LEAF ANALYZER =====")
    print(f"Ảnh cần xử lý: {len(images)}")

    for img in images:

        h, d, w = analyze_image(
            img
        )

        total_h += h
        total_d += d
        total_w += w

        print(
            f"{os.path.basename(img)}"
            f"  H={h}"
            f" D={d}"
            f" W={w}"
        )

    total = (
        total_h +
        total_d +
        total_w
    )

    adjust = 0

    if total > 0:

        dry_ratio = (
            total_d / total
        )

        wet_ratio = (
            total_w / total
        )

        # >40% lá khô
        if dry_ratio >= 0.40:

            adjust = 10

        # >40% lá ướt
        elif wet_ratio >= 0.40:

            adjust = -10

    result = {

        "time":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "images":
            len(images),

        "healthy":
            total_h,

        "dry":
            total_d,

        "wet":
            total_w,

        "adjust_percent":
            adjust
    }

    Path(
        FEEDBACK_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULT_FILE,
        "w",
        encoding="utf8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    print()
    print(
        f"Healthy : {total_h}"
    )

    print(
        f"Dry     : {total_d}"
    )

    print(
        f"Wet     : {total_w}"
    )

    print(
        f"Adjust  : {adjust}%"
    )

    return result


# =====================================================
# CẬP NHẬT THỜI GIAN TƯỚI
# =====================================================

def update_irrigation_time():

    Path(
        FEEDBACK_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        LAST_IRRIGATION,
        "w",
        encoding="utf8"
    ) as f:

        f.write(
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        )


# =====================================================
# LẤY % ĐIỀU CHỈNH
# =====================================================

def get_adjust_percent():
    Path(
        IMAGE_DIR
    ).mkdir(
        parents=True,
        exist_ok=True
    )
    if not os.path.exists(
        RESULT_FILE
    ):
        return 0

    try:

        with open(
            RESULT_FILE,
            "r",
            encoding="utf8"
        ) as f:

            data = json.load(f)

        return int(
            data.get(
                "adjust_percent",
                0
            )
        )

    except:

        return 0


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":

    result = evaluate_leaf()

    print()
    print("===== RESULT =====")
    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ))
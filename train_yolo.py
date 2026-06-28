"""
train_yolo.py
"""

import os
import shutil
import logging
import torch

from ultralytics import YOLO
from pathlib import Path
from datetime import datetime
YOLO_LOG_DIR = "FullDataset/logs/log_yolo"
GRAPH_DIR = "FullDataset/graph/yolo"
Path(GRAPH_DIR).mkdir(
    parents=True,
    exist_ok=True
)
Path(YOLO_LOG_DIR).mkdir(
    parents=True,
    exist_ok=True
)

YOLO_LOG_FILE = (
    f"{YOLO_LOG_DIR}/"
    f"yolo_train_{datetime.now().strftime('%Y%m%d')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(
            YOLO_LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# ==========================================================
# CONFIG
# ==========================================================

DATA_YAML = "FullDataset/pepper_bell/data.yaml"

PRETRAINED_MODEL = "yolov8n.pt"

MODEL_SAVE = "FullModel/modelyolo/yolo_leaf.pt"

EPOCHS = 100
IMG_SIZE = 640
BATCH_SIZE = 4

if torch.cuda.is_available():
    DEVICE = 0
    print(f"\n🚀 GPU: {torch.cuda.get_device_name(0)}\n")
else:
    DEVICE = "cpu"
    print("\n⚠️  Đang dùng CPU\n")

# ==========================================================
# TRAIN
# ==========================================================

def train():

    if not os.path.exists(DATA_YAML):
        logger.error(f"Không tìm thấy: {DATA_YAML}")
        return

    logger.info("===================================")
    logger.info("YOLOv8 PEPPER LEAF TRAINING")
    logger.info("===================================")
    logger.info(f"Dataset : {DATA_YAML}")
    logger.info(f"Model   : {PRETRAINED_MODEL}")
    logger.info(f"Epochs  : {EPOCHS}")
    logger.info(f"ImgSize : {IMG_SIZE}")
    logger.info(f"Batch   : {BATCH_SIZE}")
    logger.info(f"Device  : {DEVICE}")
    if DEVICE == 0:
        logger.info(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    model = YOLO(PRETRAINED_MODEL)

    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=2,
        cache=None,
        plots=True,
        patience=20,
        project="runs",
        name="pepper_leaf",
        exist_ok=True
    )

    best_model = results.save_dir / "weights" / "best.pt"
    import shutil
    from pathlib import Path

    GRAPH_DIR = "FullDataset/graph/yolo"
    Path(GRAPH_DIR).mkdir(parents=True, exist_ok=True)

    for file in [
        "results.png",
        "PR_curve.png",
        "F1_curve.png",
        "P_curve.png",
        "R_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png"
    ]:
        src = results.save_dir / file

        if src.exists():
            shutil.copy(
                str(src),
                f"{GRAPH_DIR}/{file}"
            )
    if best_model.exists():

        os.makedirs("FullModel/modelyolo", exist_ok=True)

        shutil.copy(
            str(best_model),
            MODEL_SAVE
        )
        #####
        result_dir = results.save_dir

        files = [
            "results.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png",
            "F1_curve.png",
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
        ]

        for file in files:

            src = result_dir / file

            if src.exists():

                shutil.copy(
                    str(src),
                    os.path.join(
                        GRAPH_DIR,
                        file
                    )
                )
        #####
        logger.info("")
        logger.info("===================================")
        logger.info("TRAIN HOÀN THÀNH")
        logger.info(f"Best model:")
        logger.info(MODEL_SAVE)
        logger.info("===================================")

    else:
        logger.error("Không tìm thấy best.pt")


# ==========================================================
# VALIDATE
# ==========================================================

def validate():
    with open(
        f"{GRAPH_DIR}/train_metrics.txt",
        "w",
        encoding="utf8"
    ) as f:

        f.write(
            f"Precision      : {metrics.box.mp:.4f}\n"
        )

        f.write(
            f"Recall         : {metrics.box.mr:.4f}\n"
        )

        f.write(
            f"mAP50          : {metrics.box.map50:.4f}\n"
        )

        f.write(
            f"mAP50-95       : {metrics.box.map:.4f}\n"
        )

    if not os.path.exists(MODEL_SAVE):
        logger.error("Chưa có model.")
        return

    model = YOLO(MODEL_SAVE)

    metrics = model.val(
        data=DATA_YAML
    )

    logger.info("")
    logger.info("=========== VALIDATION ===========")
    logger.info(f"Precision : {metrics.box.mp:.3f}")
    logger.info(f"Recall    : {metrics.box.mr:.3f}")
    logger.info(f"mAP50     : {metrics.box.map50:.3f}")
    logger.info(f"mAP50-95  : {metrics.box.map:.3f}")
    logger.info("==================================")


# ==========================================================
# TEST 1 IMAGE
# ==========================================================

def test_single_image(image_path):
    logger.info(
        f"Model: {MODEL_SAVE}"
    )
    if not os.path.exists(image_path):
        logger.error(f"Không tìm thấy ảnh: {image_path}")
        return

    if not os.path.exists(MODEL_SAVE):
        logger.error("Chưa train model.")
        return

    model = YOLO(MODEL_SAVE)

    results = model.predict(
        source=image_path,
        conf=0.4,
        device=DEVICE,
        save=True,
        save_crop=True,
        save_txt=True,
        project="runs",
        name="predict",
        exist_ok=True
    )
    total_leaf = 0
    for r in results:
        logger.info(f"Đã detect {len(r.boxes)} lá") 
        total_leaf += len(r.boxes)

    logger.info(f"Tổng số lá detect được: {total_leaf}")
    logger.info("Ảnh kết quả nằm tại:")
    logger.info("runs/predict/")

    import cv2
    import glob

    crop_dir = "runs/detect/crops/leaf"

    for img in glob.glob(crop_dir + "/*.jpg"):
        im = cv2.imread(img)
        im = cv2.resize(im, (256, 256))
        cv2.imwrite(img, im)
# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    import sys

    if len(sys.argv) == 1:

        train()

    elif sys.argv[1] == "validate":

        validate()

    elif sys.argv[1] == "test":

        if len(sys.argv) < 3:
            print("Ví dụ:")
            print("python train_yolo.py test predict_images/test.jpg")
        else:
            test_single_image(sys.argv[2])
    elif sys.argv[1] == "test_all":

        import shutil

        crop_dir = "runs/predict/crops/leaf"

        if os.path.exists(crop_dir):
            shutil.rmtree(crop_dir)

        folder = "FullDataset/predict_images"

        files = sorted(os.listdir(folder))

        for file in files:

            if file.lower().endswith((".jpg", ".jpeg", ".png")):

                print(f"Đang xử lý: {file}")

                test_single_image(
                    os.path.join(folder, file)
                )
    else:

        print()
        print("Train:")
        print("python train_yolo.py")
        print()
        print("Validate:")
        print("python train_yolo.py validate")
        print()
        print("Test:")
        print("python train_yolo.py test FullDataset/predict_images/test.jpg")


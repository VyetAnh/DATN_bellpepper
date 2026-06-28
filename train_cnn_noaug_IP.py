"""
train_cnn.py — Train CNN phân loại tình trạng lá Ớt Chuông.
"""

import os
import sys
from torch import nn
from torchvision.models import (
    mobilenet_v2,
    resnet50,
    vgg16
)
import sys
import shutil
import logging
import json
from pathlib import Path
from datetime import datetime
import torch
GRAPH_DIR = "FullDataset/graph/cnn_noaug"
Path(GRAPH_DIR).mkdir(
    parents=True,
    exist_ok=True
)
CNN_LOG_DIR = "FullDataset/logs/log_cnn"

Path(CNN_LOG_DIR).mkdir(
    parents=True,
    exist_ok=True
)

CNN_LOG_FILE = (
    f"{CNN_LOG_DIR}/"
    f"cnn_train_{datetime.now().strftime('%Y%m%d')}.log"
)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(
            CNN_LOG_FILE,
            encoding="utf-8"
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR    = "FullDataset/dataset_cnn"
YOLO_MODEL     = "FullModel/modelyolo/yolo_leaf.pt"
MODEL_NAME = "mobilenet"

if len(sys.argv) > 1:

    if sys.argv[1] in [
        "mobilenet",
        "resnet50",
        "vgg16"
    ]:
        MODEL_NAME = sys.argv[1]

    elif (
        sys.argv[1] == "validate"
        and len(sys.argv) > 2
        and sys.argv[2] in [
            "mobilenet",
            "resnet50",
            "vgg16"
        ]
    ):
        MODEL_NAME = sys.argv[2]

CNN_MODEL_SAVE = (
    f"FullModel/modelcnn/{MODEL_NAME}_noaug.pt"
)

GRAPH_DIR = (
    f"FullDataset/graph/cnn_noaug/{MODEL_NAME}"
)

Path(GRAPH_DIR).mkdir(
    parents=True,
    exist_ok=True
)
CLASSES        = ["dry", "healthy", "wet"]
IMG_SIZE       = 224
BATCH_SIZE     = 32
EPOCHS         = 50
LR             = 3e-4
best_epochs_no_improve = 0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# PHẦN 1: DATASET

def create_dataset_structure():
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            path = Path(f"{DATASET_DIR}/{split}/{cls}")
            path.mkdir(parents=True, exist_ok=True)

    logger.info(f"✅ Cấu trúc dataset tại: {DATASET_DIR}/")
    logger.info("   Đặt ảnh lá vào đúng thư mục:")
    for cls in CLASSES:
        logger.info(f"   {DATASET_DIR}/train/{cls}/  ← ảnh lá {cls}")


def check_dataset() -> bool:
    total = 0
    print(f"\n{'Class':<20} {'Train':>7} {'Val':>7}")
    print("-" * 36)
    for cls in CLASSES:
        n_train = len(list(Path(f"{DATASET_DIR}/train/{cls}").glob("*.jpg"))) + \
                  len(list(Path(f"{DATASET_DIR}/train/{cls}").glob("*.png")))
        n_val   = len(list(Path(f"{DATASET_DIR}/val/{cls}").glob("*.jpg"))) + \
                  len(list(Path(f"{DATASET_DIR}/val/{cls}").glob("*.png")))
        total  += n_train + n_val
        status  = "✅" if n_train >= 20 else "⚠️ "
        print(f"  {status} {cls:<18} {n_train:>7} {n_val:>7}")

    print(f"\n  Tổng: {total} ảnh")
    if total == 0:
        logger.error("❌ Chưa có ảnh! Đặt ảnh vào dataset_cnn/train/<class>/")
        return False
    if total < 80:
        logger.warning(f"⚠️  Ít ảnh ({total}). Nên có ≥ 200 ảnh cho kết quả tốt.")
    return True


def auto_split(source_dir, val_ratio=0.15, test_ratio=0.15):

    import random
    random.seed(42)

    for cls in CLASSES:
        src = Path(f"{source_dir}/{cls}")
        if not src.exists():
            continue

        imgs = list(src.glob("*.jpg")) + list(src.glob("*.png"))
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * val_ratio))
        n_test = max(1, int(len(imgs) * test_ratio))
        for i, img in enumerate(imgs):
            if i < n_val:
                split = "val"
            elif i < n_val + n_test:
                split = "test"
            else:
                split = "train"
            dst   = Path(f"{DATASET_DIR}/{split}/{cls}/{img.name}")
            shutil.copy(img, dst)

        logger.info(f"  {cls}: {len(imgs)-n_val-n_test} train | {n_val} val | {n_test} test")

    logger.info("✅ Chia train/val/test xong!")

    logger.info(
        f"GPU: {torch.cuda.get_device_name(0)}"
        if torch.cuda.is_available()
        else "CPU"
    )

# PHẦN 2: TRAIN CNN
def build_model(num_classes):

    if MODEL_NAME == "mobilenet":

        model = mobilenet_v2(weights="DEFAULT")

        in_features = model.classifier[1].in_features

        model.classifier = nn.Sequential(

            nn.Dropout(0.2),

            nn.Linear(in_features,512),

            nn.BatchNorm1d(512),

            nn.ReLU(inplace=True),

            nn.Dropout(0.4),

            nn.Linear(512,128),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(128,num_classes)

        )

    elif MODEL_NAME == "resnet50":

        model = resnet50(
            weights="DEFAULT"
        )

        in_features = (
            model.fc.in_features
        )

        model.fc = nn.Linear(
            in_features,
            num_classes
        )

    elif MODEL_NAME == "vgg16":

        model = vgg16(
            weights="DEFAULT"
        )

        in_features = (
            model.classifier[6]
            .in_features
        )

        model.classifier[6] = nn.Linear(
            in_features,
            num_classes
        )

    return model

def get_transforms():
    import torchvision.transforms as T
    train_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    val_tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    ])

    return train_tf, val_tf


def train():
    """Train CNN với transfer learning từ MobileNetV2."""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
    except ImportError:
        logger.error("❌ Cài: pip install torch torchvision")
        return

    create_dataset_structure()
    if not check_dataset():
        return

    train_tf, val_tf = get_transforms()

    train_ds = ImageFolder(f"{DATASET_DIR}/train", transform=train_tf)
    val_ds   = ImageFolder(f"{DATASET_DIR}/val",   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Kiểm tra class order
    logger.info(f"Classes: {train_ds.classes}")
    os.makedirs(
        "FullModel/modelcnn",
        exist_ok=True
    )
    with open(
        "FullModel/modelcnn/classes.json",
        "w"
    ) as f:

        json.dump(
            train_ds.classes,
            f,
            indent=2
        )
    if train_ds.classes != CLASSES:
        logger.warning(f"⚠️  Class order khác! Dataset: {train_ds.classes}")
        logger.warning(f"   Expected: {CLASSES}")

    device = torch.device(DEVICE)
    model  = build_model(len(CLASSES)).to(device)
    if MODEL_NAME == "mobilenet":
        for p in model.features.parameters():
            p.requires_grad=False

        for p in model.features[-3:].parameters():
            p.requires_grad=True

    elif MODEL_NAME == "vgg16":
        for p in model.features.parameters():
            p.requires_grad = False

    elif MODEL_NAME == "resnet50":
        for p in model.parameters():
            p.requires_grad = True

    # Class weights để xử lý imbalanced dataset
    class_counts = [len(list(Path(f"{DATASET_DIR}/train/{c}").glob("*")))
                    for c in CLASSES]
    total = sum(class_counts)
    weights = torch.tensor([total / (len(CLASSES) * c) if c > 0 else 1.0
                             for c in class_counts]).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=0.1
    )
    optimizer = optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=3e-4,
        weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_val_acc = 0.0
    history      = []
    patience = 5
    epochs_no_improve = 0
    
    logger.info(f"\n🚀 Bắt đầu train CNN — {EPOCHS} epochs")
    logger.info(f"   Model   : {MODEL_NAME}")
    logger.info(f"   Device  : {DEVICE}")
    logger.info(f"   Classes : {CLASSES}")
    logger.info(f"   Train   : {len(train_ds)} ảnh | Val: {len(val_ds)} ảnh")

    for epoch in range(EPOCHS):
        # ── Train ──────────────────────────────────────────────────────────────
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            _, predicted   = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total   += labels.size(0)

        # ── Validate ───────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs      = model(imgs)
                loss         = criterion(outputs, labels)
                val_loss    += loss.item()
                _, predicted  = outputs.max(1)
                val_correct  += predicted.eq(labels).sum().item()
                val_total    += labels.size(0)

        train_acc = train_correct / train_total * 100
        val_acc   = val_correct   / val_total   * 100
        scheduler.step(val_loss)

        history.append({
            "epoch": epoch+1,
            "train_acc": round(train_acc,2),
            "val_acc": round(val_acc,2),
            "train_loss": round(train_loss,4),
            "val_loss": round(val_loss,4)
        })

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{EPOCHS} | "
                f"Train: {train_acc:.1f}% | Val: {val_acc:.1f}%"
            )

        # Lưu model tốt nhất
        if val_acc > best_val_acc:

            best_val_acc = val_acc

            epochs_no_improve = 0

            os.makedirs("FullModel/modelcnn", exist_ok=True)

            torch.save(model, CNN_MODEL_SAVE)

        else:

            epochs_no_improve += 1
        if epochs_no_improve >= patience:

            logger.info(
                f"Early stopping tại epoch {epoch+1}"
            )

            break
    logger.info(f"\n✅ Train xong! Best val acc: {best_val_acc:.1f}%")
    logger.info(f"   Model lưu: {CNN_MODEL_SAVE}")
    import matplotlib.pyplot as plt

    epochs = [x["epoch"] for x in history]
    train_acc = [x["train_acc"] for x in history]
    val_acc = [x["val_acc"] for x in history]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc,label="Train")
    plt.plot(epochs, val_acc,label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid()

    plt.savefig(
        f"{GRAPH_DIR}/accuracy_curve.png",
        dpi=300
    )
    train_loss = [x["train_loss"] for x in history]
    val_loss = [x["val_loss"] for x in history]

    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="Train")
    plt.plot(epochs, val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.savefig(
        f"{GRAPH_DIR}/loss_curve.png",
        dpi=300
    )

    plt.close()
    plt.close()
    # Lưu history
    with open(f"{GRAPH_DIR}/train_history.json", "w") as f:
        json.dump({"history": history, "best_val_acc": best_val_acc,
                   "classes": CLASSES}, f, indent=2)
# PHẦN 3: VALIDATE & TEST

def validate():
    """Đánh giá chi tiết từng class trên tập val."""
    try:
        import torch
        from torchvision.datasets import ImageFolder
        from torch.utils.data import DataLoader
        from sklearn.metrics import classification_report, confusion_matrix
        import numpy as np
    except ImportError:
        logger.error("❌ Cài: pip install torch torchvision scikit-learn")
        return

    if not os.path.exists(CNN_MODEL_SAVE):
        logger.error(f"❌ Chưa có model: {CNN_MODEL_SAVE}")
        return

    _, val_tf = get_transforms()
    val_ds    = ImageFolder(f"{DATASET_DIR}/val", transform=val_tf)
    val_loader= DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    device    = torch.device(DEVICE)
    model = torch.load(
        CNN_MODEL_SAVE,
        map_location=device,
        weights_only=False
    )
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in val_loader:
            outputs      = model(imgs.to(device))
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=CLASSES, digits=3))

    print("Confusion Matrix:")
    cm = confusion_matrix(all_labels, all_preds)
    print(f"{'':>20}", end="")
    for c in CLASSES: print(f"{c:>16}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i]:>18}", end="")
        for v in row: print(f"{v:>16}", end="")
        print()
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6,5))
    plt.tight_layout()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=CLASSES,
        yticklabels=CLASSES
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.savefig(
        f"{GRAPH_DIR}/confusion_matrix.png",
        dpi=300
    )
    plt.close()
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASSES,
        digits=3
    )

    with open(
        f"{GRAPH_DIR}/classification_report.txt",
        "w",
        encoding="utf8"
    ) as f:
        f.write(report)
        
    from sklearn.metrics import accuracy_score

    acc = accuracy_score(
        all_labels,
        all_preds
    )
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import f1_score

    precision = precision_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="weighted"
    )

    f1 = f1_score(
        all_labels,
        all_preds,
        average="weighted"
    )
    with open(
        f"{GRAPH_DIR}/metrics.txt",
        "w",
        encoding="utf8"
    ) as f:
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {precision:.4f}\n")
        f.write(f"Recall    : {recall:.4f}\n")
        f.write(f"F1-score  : {f1:.4f}\n")
    

def test_image(image_path: str):

    import cv2
    import torch
    import torchvision.transforms as T
    from PIL import Image

    if not os.path.exists(image_path):
        logger.error(f"❌ Không tìm thấy ảnh: {image_path}")
        return

    logger.info(f"\n🔬 Test ảnh: {image_path}")

    # ── Bước 1: YOLO crop lá ─────────────────────────────────────────────────
    crops = []
    crop_info = []

    if os.path.exists(YOLO_MODEL):
        from ultralytics import YOLO
        yolo    = YOLO(YOLO_MODEL)
        results = yolo(image_path, conf=0.4, verbose=False)
        img_orig= cv2.imread(image_path)

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                logger.warning("⚠️  YOLO không detect lá nào → dùng toàn ảnh")
                crops.append(img_orig)
                crop_info.append({"id": 1, "bbox": "full"})
            else:
                for i, box in enumerate(boxes):
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    h, w = img_orig.shape[:2]
                    x1=max(0,x1-5); y1=max(0,y1-5)
                    x2=min(w,x2+5); y2=min(h,y2+5)
                    crop = img_orig[y1:y2, x1:x2]
                    crop = cv2.resize(
                        crop,
                        (224,224)
                    )
                    if crop.size > 0:
                        crops.append(crop)
                        crop_info.append({"id": i+1, "bbox": f"({x1},{y1},{x2},{y2})"})
        logger.info(f"🔍 YOLO detect: {len(crops)} lá")
    else:
        logger.warning(f"⚠️  Chưa có YOLO model ({YOLO_MODEL}) → dùng toàn ảnh")
        crops.append(cv2.imread(image_path))
        crop_info.append({"id": 1, "bbox": "full"})

    # ── Bước 2: CNN classify từng lá ─────────────────────────────────────────
    if not os.path.exists(CNN_MODEL_SAVE):
        logger.error(f"❌ Chưa có CNN model: {CNN_MODEL_SAVE}")
        return

    device = torch.device(DEVICE)
    model  = torch.load(CNN_MODEL_SAVE, map_location=device)
    model.eval()

    tf = T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    results_list = []
    class_counts = {c: 0 for c in CLASSES}

    print(f"\n{'Lá':>5} {'Kết quả':>16} {'Conf':>7}  dry   wet   healthy")
    print("-" * 65)

    for crop, info in zip(crops, crop_info):
        # BGR → RGB → PIL
        pil_img  = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        tensor   = tf(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            import torch.nn.functional as F
            logits  = model(tensor)
            probs   = F.softmax(logits, dim=1)[0].tolist()

        pred_cls = CLASSES[probs.index(max(probs))]
        conf     = max(probs)
        class_counts[pred_cls] += 1

        results_list.append({
            "leaf_id": info["id"],
            "prediction": pred_cls,
            "confidence": round(conf, 3),
            "dry": round(probs[0], 3),
            "wet": round(probs[1], 3),
            "healthy": round(probs[2], 3),
        })

        print(
            f"  Lá {info['id']:>2}  {pred_cls:>16}  {conf:>6.1%}  "
            f"{probs[0]:>5.1%} {probs[1]:>5.1%} {probs[2]:>8.1%}"
        )

        # Lưu crop với label
        base = os.path.splitext(image_path)[0]
        out  = f"{base}_leaf{info['id']}_{pred_cls}.jpg"
        cv2.imwrite(out, crop)

    # ── Bước 3: Tổng hợp kết quả ─────────────────────────────────────────────
    n = max(len(crops), 1)
    healthy_pct    = class_counts["healthy"]        / n * 100
    dry_pct        = class_counts["dry"]            / n * 100
    wet_pct        = class_counts["wet"]            / n * 100

    print(f"\n{'='*55}")
    print(f"📊 TỔNG HỢP ({n} lá):")
    print(f"   healthy       : {healthy_pct:.0f}%")
    print(f"   dry           : {dry_pct:.0f}%")
    print(f"   wet           : {wet_pct:.0f}%")


    # Tính điều chỉnh volume
    from leaf_analyzer import _compute_volume_adjustment
    adj = _compute_volume_adjustment(dry_pct, wet_pct, healthy_pct)
    print(f"\n💧 Điều chỉnh volume tưới: {adj:+.0f}%")

    print(f"{'='*55}")


def crop_only(image_path: str):
    """Chỉ chạy YOLO crop lá, lưu ảnh crop ra thư mục."""
    import cv2

    if not os.path.exists(YOLO_MODEL):
        logger.error(f"❌ Chưa có YOLO model: {YOLO_MODEL}")
        return
    if not os.path.exists(image_path):
        logger.error(f"❌ Không tìm thấy ảnh: {image_path}")
        return

    from ultralytics import YOLO
    yolo     = YOLO(YOLO_MODEL)
    img_orig = cv2.imread(image_path)
    results  = yolo(image_path, conf=0.4, verbose=False)

    out_dir  = os.path.splitext(image_path)[0] + "_crops"
    Path(out_dir).mkdir(exist_ok=True)
    n = 0

    for result in results:
        if result.boxes is None: continue
        for i, box in enumerate(result.boxes):
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            h,w = img_orig.shape[:2]
            x1=max(0,x1-5); y1=max(0,y1-5)
            x2=min(w,x2+5); y2=min(h,y2+5)
            crop = img_orig[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(
                    crop,
                    (224,224)
                )

                cv2.imwrite(
                    f"{out_dir}/leaf_{i+1}.jpg",
                    crop
                )
                n += 1

    logger.info(f"✅ Crop xong: {n} lá → {out_dir}/")
# MAIN

if __name__ == "__main__":

    if len(sys.argv) == 1:

        train()

    elif sys.argv[1] in [
        "mobilenet",
        "resnet50",
        "vgg16"
    ]:

        train()

    elif sys.argv[1] == "check":

        check_dataset()

    elif sys.argv[1] == "validate":

        if len(sys.argv) > 2:
            MODEL_NAME = sys.argv[2]

            CNN_MODEL_SAVE = (
                f"FullModel/modelcnn/{MODEL_NAME}_noaug.pt"
            )

            GRAPH_DIR = (
                f"FullDataset/graph/cnn_noaug/{MODEL_NAME}"
            )

            Path(GRAPH_DIR).mkdir(
                parents=True,
                exist_ok=True
            )

        validate()

    else:
        print("""
        LỖI
        """)

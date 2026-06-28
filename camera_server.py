"""
camera_server.py — Nhận ảnh từ ESP32-CAM qua HTTP POST (ngrok tunnel).
Chạy song song với main.py bằng thread riêng.

ESP32-CAM + A7680C gửi ảnh lên:
    POST https://xxxx.ngrok.io/upload
    Body: multipart/form-data, field "image"

Server lưu ảnh → gọi leaf_analyzer → cập nhật leaf_score toàn cục
"""

import os
import logging
import threading
from datetime import datetime
from pathlib import Path
from flask import Flask, request, jsonify

from config import *
from leaf_analyzer import analyze_image, get_latest_leaf_score

logger = logging.getLogger(__name__)

app = Flask(__name__)

# ── Thư mục lưu ảnh ───────────────────────────────────────────────────────────
IMAGE_DIR = "data/images"
Path(IMAGE_DIR).mkdir(parents=True, exist_ok=True)


@app.route("/health", methods=["GET"])
def health():
    """Health check — ESP32 ping trước khi gửi ảnh."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


@app.route("/upload", methods=["POST"])
def upload_image():
    """
    Nhận ảnh từ ESP32-CAM.
    ESP32 gửi: POST /upload với field "image" là file ảnh JPEG.
    """
    if "image" not in request.files:
        return jsonify({"error": "Không tìm thấy field 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "File rỗng"}), 400

    # Lưu ảnh với timestamp
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"leaf_{ts}.jpg"
    filepath = os.path.join(IMAGE_DIR, filename)
    file.save(filepath)

    logger.info(f"📸 Nhận ảnh: {filename} ({os.path.getsize(filepath)/1024:.1f} KB)")

    # Phân tích lá ngay sau khi nhận ảnh
    try:
        result = analyze_image(filepath)
        logger.info(
            f"🌿 Leaf analysis: "
            f"healthy={result['healthy_pct']:.0f}% | "
            f"dry={result['dry_pct']:.0f}% | "
            f"wet={result['wet_pct']:.0f}% | "
            f"bacterial={result['bacterial_pct']:.0f}% | "
            f"adjustment={result['volume_adjustment_pct']:+.0f}%"
        )
        if result["bacterial_alert"]:
            logger.warning(f"⚠️  CẢNH BÁO: {result['bacterial_alert']}")

        return jsonify({
            "status":      "ok",
            "filename":    filename,
            "leaf_result": result,
        })
    except Exception as e:
        logger.error(f"❌ Lỗi phân tích ảnh: {e}")
        return jsonify({"status": "saved", "filename": filename, "error": str(e)}), 200


@app.route("/leaf_score", methods=["GET"])
def leaf_score_endpoint():
    """Lấy leaf_score hiện tại (để debug hoặc app đọc)."""
    return jsonify(get_latest_leaf_score())


def run_camera_server(host: str = "0.0.0.0", port: int = 5001):
    """Chạy Flask server trong thread riêng."""
    logger.info(f"📡 Camera server khởi động: http://{host}:{port}")
    logger.info(f"   ESP32-CAM gửi ảnh lên: POST /upload")
    logger.info(f"   Dùng ngrok: ngrok http {port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)


def start_camera_server_thread(port: int = 5001):
    """Khởi động camera server trong background thread."""
    t = threading.Thread(
        target=run_camera_server,
        kwargs={"port": port},
        daemon=True
    )
    t.start()
    logger.info(f"✅ Camera server thread started (port {port})")
    return t

"""
main.py — Điểm khởi động chính.
"""

import os
import sys
import logging
import schedule
import time
import json
from datetime import datetime
from pathlib import Path
from leaf_analyzer import capture_if_needed
from config import *
from collector import init_firebase, fetch_latest_sensor, append_to_csv, get_csv_stats
from trainer import train_once
from mqtt_handler import send_command_mqtt
# Logging 
Path(LOG_DIR).mkdir(
    parents=True,
    exist_ok=True
)

Path(DATA_DIR).mkdir(
    parents=True,
    exist_ok=True
)

Path(MODEL_DIR).mkdir(
    parents=True,
    exist_ok=True
)

log_file = f"{LOG_DIR}/system_{datetime.now().strftime('%Y%m%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

# Load models 1 lần khi khởi động
reg_soil_model = None
reg_vol_model = None

def load_models_if_exist():
    global reg_soil_model
    global reg_vol_model

    soil_path = f'{MODEL_DIR}/rf_regressor_soil.pkl'
    vol_path  = f'{MODEL_DIR}/rf_regressor_volume.pkl'

    if not os.path.exists(vol_path):
        return False

    import joblib

    reg_vol_model = joblib.load(vol_path)

    if os.path.exists(soil_path):
        reg_soil_model = joblib.load(soil_path)
        logger.info("✅ Soil model loaded")

    logger.info("✅ Volume model loaded")

    return True


def job_collect_and_predict():
    """
    1 chu kỳ 5 phút:
    Đọc Firebase → Dự đoán AI → In kết quả → Ghi CSV → Gửi MQTT → ESP32 đọc điều khiển relay
    """
    # 1. Đọc cảm biến
    sensor = fetch_latest_sensor()
    if not sensor:
        logger.warning("⚠️  Bỏ qua chu kỳ này — không có dữ liệu cảm biến")
        return

    # 2. Dự đoán AI
    if reg_vol_model is None:
        logger.warning("⚠️ Chưa có model — dùng FAO-56")
        result = _fao_rule_based(sensor)
    else:
        from predictor import predict
        result = predict(sensor, reg_soil_model, reg_vol_model)

    # Leaf Feedback
    if result["pump_on"]:

        from leaf_analyzer import (
            evaluate_leaf,
            update_irrigation_time
        )

        feedback = evaluate_leaf()

        adjust = feedback["adjust_percent"]

        old_volume = result["volume_mL"]

        result["volume_mL"] = round(
            old_volume * (1 + adjust / 100),
            2
        )

        result["volume_L"] = round(
            result["volume_mL"] / 1000,
            5
        )

        result["relay_on_ms"] = int(
            result["volume_L"]
            / PUMP_FLOW_LPS
            * 1000
        )

        result["duration_s"] = round(
            result["relay_on_ms"] / 1000,
            2
        )

        logger.info(
            f"🌿 Leaf Feedback: {adjust:+d}% | "
            f"{old_volume:.2f} -> {result['volume_mL']:.2f} mL"
        )

    # 3. In kết quả
    _print_result(sensor, result)

    # 4. Ghi CSV
    append_to_csv(sensor, result)

    # 5. MQTT
    send_command_mqtt(result)
    
    from leaf_analyzer import update_irrigation_time

    if result["pump_on"] == 1:
        update_irrigation_time()

    # Nếu có tưới thì cập nhật thời điểm tưới
    if result["pump_on"]:
        update_irrigation_time()

    # Thống kê CSV
    stats = get_csv_stats()
    logger.info(f"📊 CSV: {stats['rows']:,} hàng | {stats['size_mb']} MB")


def _print_result(sensor: dict, result: dict):
    """In kết quả dự đoán ra terminal."""
    ts   = sensor.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    soil = sensor.get('soil_sensor_pct', '?')
    temp = sensor.get('temp_sensor_c', '?')
    hour = sensor.get('hour', datetime.now().hour)

    print()
    print("┌─────────────────────────────────────────────────┐")
    print(f"│  🌶️  {ts}")
    print(f"│  Cảm biến: Soil={soil}%  Temp={temp}°C  Giờ={hour}h")
    print(f"│  FAO Soil:{result.get('soil_pred_6h','?')}%")
    print(f"│  RF Soil : "f"{result.get('soil_future_pred','?')}%")
    print("├─────────────────────────────────────────────────┤")
    if result['pump_on']:
        print(f"│  ✅ BẬT BƠM")
        print(f"│     relay_on_ms : {result['relay_on_ms']} ms")
        print(f"│     volume_mL   : {result['volume_mL']} mL")
        print(f"│     duration_s  : {result.get('duration_s', '?')} s")
    else:
        print(f"│  ⛔ TẮT BƠM")

        if result.get('override_reason'):
            print(f"│     Lý do: {result['override_reason']}")
        elif result.get('is_hot_sun'):
            print(f"│     Lý do: Nắng gắt (10h–15h)")
        elif not result.get('is_good_hour'):
            print(f"│     Lý do: Ngoài khung giờ tưới")
        else:
            print(f"│     Lý do: Đất chưa cần tưới")
    print(f"│  FAO-56: ET₀={result.get('ET0_mm_day','?')}mm | "
          f"VPD={result.get('vpd_kPa','?')}kPa | "
          f"Ks={result.get('Ks','?')}")
    print("└─────────────────────────────────────────────────┘")


def _fao_rule_based(sensor: dict) -> dict:
    """
    Fallback khi chưa có model: dùng FAO-56 rule-based đơn giản.
    """
    from features import compute_features
    from labels import compute_fao56_labels
    import pandas as pd

    row = dict(sensor)
    row.setdefault('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    row.setdefault('pump_on', 0)
    df_in  = pd.DataFrame([row])
    df_f   = compute_features(df_in)
    df_lbl = compute_fao56_labels(df_f)
    r      = df_lbl.iloc[-1]
    vol_L  = float(r['irrigation_volume_L'])

    if vol_L > 0:
        dur_s = vol_L / PUMP_FLOW_LPS
    else:
        dur_s = 0.0

    return {
        'pump_on':      int(r['pump_on']),
        'xac_suat':     1.0 if r['pump_on'] else 0.0,
        'relay_on_ms':  int(dur_s * 1000),
        'duration_s':   round(dur_s, 3),
        'volume_mL':    round(vol_L * 1000, 2),
        'volume_L':     round(vol_L, 5),
        'soil_future_pred': round(float(r.get('soil_pred_6h', 0)), 1),
        'soil_pred_6h': round(float(r.get('soil_pred_6h', 0)), 1),
        'hours_to_dry': round(float(r.get('hours_to_dry', 0)), 1),
        'ET0_mm_day':   round(float(r.get('ET0_mm_day', 0)), 3),
        'ETc_adj':      round(float(r.get('ETc_adj_mm_day', 0)), 3),
        'Kc':           round(float(r.get('Kc', 0)), 3),
        'Ks':           round(float(r.get('Ks', 0)), 3),
        'vpd_kPa':      round(float(r.get('vpd_kPa', 0)), 3),
        'is_hot_sun':   int(r.get('is_hot_sun', 0)),
        'is_good_hour': int(r.get('is_good_hour', 0)),
        'IR_adj':       round(float(r.get('IR_adj', 0)), 4),
    }


def job_train():
    """Train lại models, reload sau khi xong."""
    try:
        results = train_once()
        if results:
            # Reload models mới vào bộ nhớ
            load_models_if_exist()
            logger.info(
                f"✅ Train xong & reload volume model: "
                f"Soil MAE={results.get('soil_mae')}% | "
                f"Volume MAE={results.get('volume_mae')}L"
            )
    except Exception as e:
        logger.error(f"❌ Lỗi train: {e}")


def setup_schedule():
    schedule.every(COLLECT_INTERVAL_MIN).minutes.do(job_collect_and_predict)
    logger.info(f"⏰ Thu thập + dự đoán: mỗi {COLLECT_INTERVAL_MIN} phút")

    if TRAIN_INTERVAL_DAYS == 1:
        schedule.every().day.at(TRAIN_TIME).do(job_train)
        logger.info(f"⏰ Train: mỗi ngày lúc {TRAIN_TIME}")
    else:
        schedule.every(TRAIN_INTERVAL_DAYS).days.at(TRAIN_TIME).do(job_train)
        logger.info(f"⏰ Train: mỗi {TRAIN_INTERVAL_DAYS} ngày lúc {TRAIN_TIME}")


def main():
    print()
    logger.info("=" * 60)
    logger.info("🌶️  IRRIGATION AI SYSTEM — Ớt Chuông")
    logger.info(f"   Ngày trồng  : {PLANTING_DATE}")
    logger.info(f"   Vùng rễ     : {ROOT_VOL_L*1000:.0f}cm³")
    logger.info(f"   Soil target : {SOIL_TARGET}% (không tưới full {SOIL_SENSOR_MAX}%)")
    logger.info(f"   Thu thập    : mỗi {COLLECT_INTERVAL_MIN} phút")
    logger.info(f"   Train       : mỗi {TRAIN_INTERVAL_DAYS} ngày lúc {TRAIN_TIME}")
    logger.info(f"   CSV         : {CSV_RAW}")
    logger.info(f"   Log         : {log_file}")
    logger.info("=" * 60)
    logger.info("📦 Cột CSV: cảm biến + [pump_on | relay_on_ms | volume_mL]")
    logger.info("🔼 MQTT: bellpepper/command → ESP32 đọc điều khiển relay")
    logger.info("=" * 60)

    # Khởi tạo Firebase
    try:
        init_firebase()
    except Exception as e:
        logger.error(f"❌ Firebase init thất bại: {e}")
        logger.error("   Kiểm tra FIREBASE_KEY_PATH và FIREBASE_DATABASE_URL trong config.py")
        sys.exit(1)

    # Load models nếu đã có
    has_model = load_models_if_exist()

    # Train ngay nếu chưa có model
    if not has_model:
        logger.info("🤖 Chưa có model — thu thập trước, train sau khi đủ dữ liệu...")
        logger.info("   (Đang dùng FAO-56 rule-based tạm thời)")
    else:
        results_file = f'{MODEL_DIR}/train_results.json'
        if os.path.exists(results_file):
            with open(results_file) as f:
                r = json.load(f)

            logger.info(
                f"✅ Model sẵn sàng — train lần cuối: "
                f"{r.get('trained_at')} | "
                f"Volume MAE={r.get('volume_mae')}L | "
                f"Soil MAE={r.get('soil_mae')}% | "
                f"{r.get('n_samples')} mẫu"
            )

    # Chạy ngay lần đầu
    logger.info("🔄 Chạy lần đầu...")
    capture_if_needed()
    job_collect_and_predict()

    # Thiết lập lịch
    setup_schedule()

    # Vòng lặp chính
    logger.info("▶️  Hệ thống đang chạy... (Ctrl+C để dừng)")
    try:
        while True:
            capture_if_needed()
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("⏹️  Dừng hệ thống.")


if __name__ == "__main__":
    main()

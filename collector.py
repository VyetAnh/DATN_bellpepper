"""
collector.py — Thu thập dữ liệu từ Firebase mỗi 5 phút, ghi vào CSV.
"""

import os
import json
import logging
import pandas as pd
import requests
from datetime import datetime
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, db

from config import *

logger = logging.getLogger(__name__)


def init_firebase():
    """Khởi tạo Firebase Admin SDK."""
    if not firebase_admin._apps:
        if os.path.exists(FIREBASE_KEY_PATH):
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
        else:
            # Đọc từ biến môi trường (khi deploy)
            key_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
            if not key_json:
                raise FileNotFoundError(
                    f"Không tìm thấy {FIREBASE_KEY_PATH} và biến môi trường FIREBASE_SERVICE_ACCOUNT_KEY"
                )
            cred = credentials.Certificate(json.loads(key_json))

        firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})
        logger.info("✅ Firebase initialized")



def get_weather_data():

    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={LATITUDE}"
        f"&longitude={LONGITUDE}"
        f"&current=temperature_2m,"
        f"relative_humidity_2m,"
        f"surface_pressure,"
        f"wind_speed_10m"

        f"&hourly=temperature_2m,"
        f"relative_humidity_2m,"
        f"precipitation_probability,"
        f"shortwave_radiation,"
        f"precipitation"
        f"&forecast_days=1"

        f"&timezone=Asia/Ho_Chi_Minh"
        f"&cell_selection=land"

        f"&temperature_unit=celsius"
        f"&wind_speed_unit=ms"

        f"&models=gem_seamless"
    )
    response = requests.get(
        url,
        timeout=10
    )

    data = response.json()

    current = data["current"]
    hourly = data["hourly"]

    times = hourly["time"]

    # tạo thời gian hiện tại đúng timezone VN
    current_time = datetime.now().strftime(
        "%Y-%m-%dT%H:00"
    )

    try:
        idx = times.index(current_time)
    except ValueError:
        idx = datetime.now().hour

    # gió ở độ cao 10m
    wind_10m = current["wind_speed_10m"]

    # hiệu chỉnh về gần mặt đất
    wind_surface = wind_10m * 0.75

    return {

        "solar_radiation":
            hourly["shortwave_radiation"][idx],

        "wind_speed":
            round(wind_surface, 2),
        "rain_1h_mm":
            hourly["precipitation"][idx],

        "rain_3h_mm":
            sum(hourly["precipitation"][idx:idx+3]),

        "rain_6h_mm":
            sum(hourly["precipitation"][idx:idx+6]),

        "rain_probability_1h":
            hourly["precipitation_probability"][idx],

        "rain_probability_3h":
            max(hourly["precipitation_probability"][idx:idx+3]),

        "rain_probability_6h":
            max(hourly["precipitation_probability"][idx:idx+6]),

        "forecast_temp_1h":
            hourly["temperature_2m"][min(idx+1,23)],
        
        "forecast_temp_3h":
            hourly["temperature_2m"][min(idx+3,23)],

        "forecast_temp_6h":
            hourly["temperature_2m"][min(idx+6,23)],

        "forecast_humidity_3h":
            hourly["relative_humidity_2m"][min(idx+3,23)],

        "forecast_humidity_6h":
            hourly["relative_humidity_2m"][min(idx+6,23)],
            
        "forecast_solar_3h":
            hourly["shortwave_radiation"][min(idx+3,23)],

        "forecast_solar_6h":
            hourly["shortwave_radiation"][min(idx+6,23)],
    }
def fetch_latest_sensor():

    try:
        ref = db.reference("/He_thong_tuoi/sensors/data")
        data = ref.get()

        if not data:
            return None

        keys = sorted(
            data.keys(),
            key=lambda x: int(x)
        )

        latest = None

        # duyệt từ cuối lên đầu
        for key in reversed(keys):

            item = data[key]

            # bỏ dòng lỗi
            if (
                item.get("temperature", 0) in [0, None]
                or item.get("humidity", 0) in [0, None]
                or item.get("light_lux", 0) in [-1, -2, None]
                or item.get("datetime") in [None, "", "1970-01-01 07:00:00"]
            ):

                continue

            # gặp dòng hợp lệ thì lấy
            latest = item

            break


        if latest is None:

            logger.warning(
                "⚠️ Không có dữ liệu hợp lệ"
            )

            return None

        # tính tuổi cây từ config.py
        planting_date = datetime.strptime(
            PLANTING_DATE,
            "%Y-%m-%d"
        )

        plant_age = (
            datetime.now() - planting_date
        ).days
        # ===== OPEN METEO =====
        weather = get_weather_data()

        sensor_data = {

            # Firebase
            "timestamp": latest.get("datetime"),
            # cây
            "plant_age_days": plant_age,            
            "temp_sensor_c": latest.get("temperature"),
            "humidity_sensor_pct": latest.get("humidity"),
            "pressure_hpa": latest.get("pressure_hpa"),
            "lux_bh1750": latest.get("light_lux"),

            "soil_sensor_pct": latest.get("soil_percent"),



            # OpenMeteo
            "solar_radiation_wm2": weather["solar_radiation"],
            "wind_speed_ms": weather["wind_speed"],

            "rain_6h_mm": weather["rain_6h_mm"],

            #"flow_L_min": latest.get("flow_L_min"),
            #"total_volume_L": latest.get("total_volume_L"),
            
            
        }

        return sensor_data

    except Exception as e:
        logger.error(f"❌ Firebase error: {e}")
        return None


def append_to_csv(sensor_data: dict, result: dict):
    """
    Thêm 1 dòng dữ liệu vào CSV.
    Tự động thêm timestamp nếu chưa có.
    = toàn bộ dữ liệu cảm biến + 3 cột kết quả AI:
        pump_on, relay_on_ms, volume_mL
    """
    Path(DATA_DIR).mkdir(
        parents=True,
        exist_ok=True
    )

    # Đảm bảo có timestamp
    if 'timestamp' not in sensor_data:
        sensor_data['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    
    # Ghép cảm biến + 3 cột kết quả AI
    row = {
        **sensor_data,
        "pump_on":     result["pump_on"],
        #"relay_on_ms": result["relay_on_ms"],
        "volume_mL":   result["volume_mL"],
    }

    df_new = pd.DataFrame([row])
    # thứ tự cột cố định
    fixed_cols = [

        "timestamp",
        "plant_age_days",

        "temp_sensor_c",
        "humidity_sensor_pct",
        "pressure_hpa",
        "lux_bh1750",
        "soil_sensor_pct",

        "solar_radiation_wm2",
        "wind_speed_ms",

        "rain_6h_mm",

        "pump_on",
        #"relay_on_ms",
        "volume_mL"
    ]

    # tạo cột thiếu
    for col in fixed_cols:
        if col not in df_new.columns:
            df_new[col] = None

    # ép đúng thứ tự
    df_new = df_new[fixed_cols]

    # ghi file
    if os.path.exists(CSV_RAW):
        df_new.to_csv(
            CSV_RAW,
            mode='a',
            header=False,
            index=False
        )
    else:
        df_new.to_csv(
            CSV_RAW,
            index=False
        )

        logger.info(f"✅ Tạo file CSV mới: {CSV_RAW}")

    logger.info("📥 Ghi đầy đủ dữ liệu:")

    for key, value in sensor_data.items():
        logger.info(f"   {key}: {value}")


def collect_once():
    logger.info("🔄 Thu thập dữ liệu Firebase...")

    data = fetch_latest_sensor()

    print("\n===== DATA NHẬN ĐƯỢC =====")
    print(data)
    print("==========================\n")

    return data is not None

def get_csv_stats() -> dict:
    """Thống kê CSV hiện tại."""

    if not os.path.exists(CSV_RAW):
        return {
            "rows": 0,
            "exists": False
        }

    try:
        # bỏ qua các dòng lỗi nếu số cột bị lệch
        df = pd.read_csv(
            CSV_RAW,
            on_bad_lines='skip'
        )

        return {
            "exists": True,
            "rows": len(df),
            "columns": df.shape[1],

            "from":
                df['timestamp'].iloc[0]
                if len(df) > 0 else None,

            "to":
                df['timestamp'].iloc[-1]
                if len(df) > 0 else None,

            "size_mb":
                round(
                    os.path.getsize(CSV_RAW)
                    /1024/1024,
                    2
                )
        }

    except Exception as e:

        logger.error(
            f"❌ Lỗi đọc CSV: {e}"
        )

        return {
            "exists": True,
            "rows": 0,
            "columns": 0,
            "error": str(e)
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    init_firebase()

    collect_once()

    print(
        get_csv_stats()
    )
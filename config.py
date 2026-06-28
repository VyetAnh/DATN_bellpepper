# config.py — Cấu hình toàn bộ hệ thống tưới Ớt Chuông

import math
# Ngày trồng cây (ESP32 tự tính plant_age_days từ đây) 
PLANTING_DATE = "2026-05-15"   # định dạng YYYY-MM-DD

# Firebase 
FIREBASE_DATABASE_URL = "https://precise-irrigation-6c076-default-rtdb.firebaseio.com"
FIREBASE_KEY_PATH     = "serviceAccountKey.json"   # đường dẫn đến file service account key
FIREBASE_SENSOR_PATH  = "/He_thong_tuoi/sensor/device1/data"      # node ESP32 ghi dữ liệu lên
#FIREBASE_COMMAND_PATH = "/irrigation_command" # node server ghi lệnh xuống
FIREBASE_HISTORY_PATH = "/irrigation_history"

# Đường dẫn file 
DATA_DIR        = "FullDataset/data"
CSV_RAW         = "FullDataset/data/data_raw.csv"          # dữ liệu thô thu thập từ Firebase
CSV_LABELED     = "FullDataset/data/data_labeled.csv"      # dữ liệu đã có nhãn FAO-56/thực tế
MODEL_DIR       = "FullModel/models"
LOG_DIR         = "FullDataset/logs/log_system"

# Lịch chạy ─
COLLECT_INTERVAL_MIN  = 5      # thu thập Firebase mỗi N phút
TRAIN_TIME            = "02:00"  # giờ train mỗi ngày (2h sáng)
TRAIN_INTERVAL_DAYS   = 5      # train mỗi N ngày (đổi thành 3 hoặc 5 khi cần)

# Bơm 
PUMP_FLOW_LPS = 0.2          # lưu lượng bơm (L/s)

#  Chậu & Cảm biến 
#
#  Chậu: 12×12×9 cm
#  Cảm biến: cắm sâu 7cm (đo vùng đất 7cm trên)
#
#  Tính lượng nước theo 12×12×7cm vì:
#  - Cảm biến chỉ đo đến 7cm → soil% đại diện vùng đất 0–7cm
#  - 2cm đáy (7–9cm) cảm biến không đo được → không tính vào
#  - Tưới đủ thấm 7cm là cảm biến sẽ đọc đúng
#
POT_W, POT_L, POT_D  = 10, 10, 9       # cm — kích thước chậu thực tế
SENSOR_DEPTH_CM       = 7.0            # cm — độ sâu cảm biến
POT_VOL_L             = POT_W * POT_L * POT_D / 1000           
WATER_CALC_VOL_L      = POT_W * POT_L * SENSOR_DEPTH_CM / 1000 

COIR_WATER_RETENTION  = 0.75   # sơ dừa giữ 75% nước


# Tham khảo
ROOT_VOL_L   = 3 * 3 * 6 / 1000                  # 0.054 L
SENSOR_VOL_L = math.pi * 3.5**2 * 7.0 / 1000     # 0.269 L



# Thể tích tham khảo (không dùng để tính nước)
SENSOR_VOL_L = math.pi * 3.5**2 * 7.0 / 1000   # ~0.269 L (chỉ dùng debug)

ROOT_VOL_L   = 3 * 3 * 6 / 1000                  # 0.054 L (chỉ dùng debug)

# Soil sensor 
#   0%   = khô hoàn toàn
#   60%  = bắt đầu hơi khô
#  <65%  = cần tưới (dưới vùng ổn định)
#  65–80%= vùng ổn định tốt nhất
#   100%  = bão hoà (không tưới thêm → tràn)
#  >100%  = tràn nước ra đáy chậu
SOIL_SENSOR_MAX  = 100.0
SOIL_OPTIMAL_HI  = 80.0   # trần vùng ổn định — không tưới khi ≥ đây
SOIL_OPTIMAL_LO  = 65.0   # sàn vùng ổn định — dưới đây cần tưới
SOIL_TARGET      = 75.0   # mức đích khi tưới (giữa vùng tối ưu)
SOIL_WARNING     = 70.0   # bắt đầu tưới sáng
SOIL_RAW_S       = 60.0   # RAW — tưới ngay
SOIL_WP_S        = 40.0   # Wilting Point — nguy hiểm

# FAO-56 scale
SOIL_FC_R   = SOIL_OPTIMAL_HI / SOIL_SENSOR_MAX * 100
SOIL_RAW_R  = SOIL_RAW_S      / SOIL_SENSOR_MAX * 100
SOIL_WP_R   = SOIL_WP_S       / SOIL_SENSOR_MAX * 100
TAW_R       = SOIL_FC_R - SOIL_WP_R
P_DEPLETION = 0.35
RAW_R       = P_DEPLETION * TAW_R

# Lọc nhiễu 
SOIL_MEDIAN_WINDOW = 5
SOIL_OUTLIER_DELTA = 8.0
DRYING_RATE_MAX    = 1.5

# Khung giờ tưới 
PRIME_HOURS   = list(range(5, 9))
SECOND_HOURS  = list(range(17, 20))
ALLOWED_HOURS = PRIME_HOURS + SECOND_HOURS
BAD_HOURS     = [h for h in range(24) if h not in ALLOWED_HOURS]

MORNING_HOURS = [5, 6, 7, 8]
EVENING_HOURS = [17, 18, 19]
# Tưới cứu buổi tối nếu dưới ngưỡng này
SOIL_EVENING_RESCUE = 64.0
# Target riêng cho tưới cứu buổi tối
SOIL_TARGET_EVENING = 70.0
# RF 
RF_N_ESTIMATORS     = 300
RF_MIN_SAMPLES_LEAF = 2
RF_RANDOM_STATE     = 42
PUMP_THRESHOLD      = 0.50

# Open-Meteo 

# Vị trí lấy thời tiết
LATITUDE  = 21.0042
LONGITUDE = 105.8431

# Tần suất cập nhật thời tiết (phút)
WEATHER_UPDATE_MIN = 5


# Các biến thời tiết cần lấy
OPENMETEO_CURRENT = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "rain",
]

OPENMETEO_HOURLY = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation_probability",
    "shortwave_radiation",
]
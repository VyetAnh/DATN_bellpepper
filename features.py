# features.py — Tính toàn bộ đặc trưng từ cảm biến ESP32 + Open-Meteo API.

import numpy as np
import pandas as pd
from config import *


def _clean_soil(soil: pd.Series) -> pd.Series:
    """
    Làm sạch nhiễu cảm biến soil:
    1. Outlier: thay đổi >SOIL_OUTLIER_DELTA% trong 1 bước → dùng giá trị trước
    2. Median filter: median 5 điểm → loại jitter ±1-2%
    """
    s = soil.copy()

    # Bước 1: Loại outlier bất thường (nhảy số đột ngột)
    for i in range(1, len(s)):
        if abs(s.iloc[i] - s.iloc[i-1]) > SOIL_OUTLIER_DELTA:
            s.iloc[i] = s.iloc[i-1]   # giữ giá trị trước

    # Bước 2: Median filter — loại jitter 81→83→81
    s_clean = s.rolling(SOIL_MEDIAN_WINDOW, min_periods=1).median()

    return s_clean

def compute_features(df: pd.DataFrame, api_available: bool = True) -> pd.DataFrame:
    """
    Tính toàn bộ features từ CSV thô.

    CSV cần có:
      ESP32 : timestamp, plant_age_days, temp_sensor_c, humidity_sensor_pct,
              pressure_hpa, lux_bh1750, soil_sensor_pct (0-100%)
      API   : solar_radiation_wm2, wind_speed_ms,
              rain_1h_mm, rain_3h_mm, rain_6h_mm,
              rain_prob_1h, rain_prob_3h, rain_prob_6h,
              forecast_temp_1h_c, forecast_temp_3h_c, forecast_temp_6h_c,
            
              forecast_solar_3h, forecast_solar_6h
    """
    d = df.copy().reset_index(drop=True)
    for col in [
        'rain_1h_mm',
        'rain_3h_mm',
        'rain_prob_1h',
        'rain_prob_3h',
        'rain_prob_6h',
    ]:
        if col not in d.columns:
            d[col] = 0
    # ── Tự tính hour, month, doy từ timestamp ────────────────────────────────
    # ── Time features ──────────────────────
    ts = pd.to_datetime(d['timestamp'])

    d['time_float'] = (
        ts.dt.hour +
        ts.dt.minute / 60.0
    ).round(2)
    d['hour_sin'] = np.sin(
        2*np.pi*d['time_float']/24
    ).round(4)

    d['hour_cos'] = np.cos(
        2*np.pi*d['time_float']/24
    ).round(4)
    d['hour'] = ts.dt.hour
    d["is_morning"] = (
        (d["hour"] >= 5) &
        (d["hour"] <= 9)
    ).astype(int)

    d["is_evening"] = (
        (d["hour"] >= 15) &
        (d["hour"] <= 20)
    ).astype(int)
    T    = d['temp_sensor_c']
    RH   = d['humidity_sensor_pct']
    P    = d['pressure_hpa']
    lux  = d['lux_bh1750']
    soil = d['soil_sensor_pct'] 
    d['month'] = ts.dt.month
    # Lọc jitter ±1-2%
    soil_clean = _clean_soil(soil)
    d['soil_clean'] = soil_clean


    d['soil_real'] = (soil_clean / SOIL_SENSOR_MAX * 100).clip(0, 100).round(2)

    # BƯỚC 1: Prev & Delta 
    for col, alias in [
        ('temp_sensor_c', 'temp'), ('humidity_sensor_pct', 'hum'),
        ('lux_bh1750', 'lux'), ('soil_sensor_pct', 'soil'), ('pressure_hpa', 'pres')
    ]:
        d[f'{alias}_prev']  = d[col].shift(1).fillna(d[col])
        d[f'delta_{alias}'] = (d[col] - d[f'{alias}_prev']).round(2)
    # Delta soil dùng soil_clean 
    d['soil_prev']   = soil_clean.shift(1).fillna(soil_clean)
    d['delta_soil']  = (soil_clean - d['soil_prev']).round(3)
    d['dry_delta_soil'] = np.minimum(
        d['delta_soil'],
        0
)
    # BƯỚC 2: BME280 — Áp suất hơi, VPD, Dew Point, Heat Index

    d['es_kPa']     = 0.6108 * np.exp((17.27 * T) / (T + 237.3))   # FAO Eq.11
    d['ea_kPa']     = d['es_kPa'] * (RH / 100.0)                    # FAO Eq.17
    d['vpd_kPa']    = (d['es_kPa'] - d['ea_kPa']).round(4)
    d['delta_kPaC'] = (4098 * d['es_kPa']) / ((T + 237.3) ** 2)     # FAO Eq.13
    d['gamma_kPaC'] = 0.000665 * (P / 10.0)                         # FAO Eq.8

    # Dew Point — Magnus / Lawrence 
    gm = (17.27 * T) / (T + 237.3) + np.log(np.clip(RH / 100.0, 1e-6, 1.0))
    d['dew_point_c']    = ((237.3 * gm) / (17.27 - gm)).round(2)
    d['temp_dew_gap_c'] = (T - d['dew_point_c']).round(2)

    # Heat Index — Steadman 
    Tf  = T * 9 / 5 + 32
    HI  = (-42.379 + 2.04901523*Tf + 10.14333127*RH
           - 0.22475541*Tf*RH - 0.00683783*Tf**2
           - 0.05481717*RH**2 + 0.00122874*Tf**2*RH
           + 0.00085282*Tf*RH**2 - 0.00000199*Tf**2*RH**2)
    d['heat_index_c']     = ((HI - 32) * 5 / 9).round(2)
    d['abs_humidity_gm3'] = (
        (6.112 * np.exp((17.67*T)/(T+243.5)) * RH * 2.1674) / (273.15 + T)
    ).round(3)
    d['pressure_trend'] = P.diff().fillna(0).round(2)

    # BƯỚC 3: BH1750 — PAR, DLI, flags
    d['PAR_umol']    = (lux * 0.0185).round(2)
    d['DLI_mol_m2']  = (d['PAR_umol'] * 3600 / 1e6).round(5)
    d['is_hot_sun']  = ((lux > 55000) & (d['hour'] >= 10) & (d['hour'] <= 15)).astype(int)
    d['is_good_hour']= d['hour'].isin(ALLOWED_HOURS).astype(int)
    d['heat_stress'] = (d['vpd_kPa'] * (lux / 70000.0)).round(4)


    # BƯỚC 4: FAO-56 — ET₀ Penman–Monteith (Eq.6)

    Rs_MJ  = d['solar_radiation_wm2'] * 0.0864
    Rns    = (1 - 0.23) * Rs_MJ                                       # Eq.38
    T_K    = T + 273.16
    Rs_Rso = np.clip(Rs_MJ / (0.75 * 40 + 1e-6), 0.2, 1.0)
    Rnl    = (4.903e-9 * T_K**4
              * (0.34 - 0.14 * np.sqrt(d['ea_kPa']))
              * (1.35 * Rs_Rso - 0.35))                                # Eq.39
    d['Rn_MJm2day'] = (Rns - Rnl).round(4)                            # Eq.40
    u2     = d['wind_speed_ms'] * (4.87 / np.log(67.8 * 1.0 - 5.42)) # Eq.47
    num    = (0.408 * d['delta_kPaC'] * d['Rn_MJm2day']
              + d['gamma_kPaC'] * (900 / (T + 273)) * u2 * d['vpd_kPa'])
    den    = d['delta_kPaC'] + d['gamma_kPaC'] * (1 + 0.34 * u2)
    d['ET0_mm_day'] = np.maximum(num / den, 0.0).round(4)             # Eq.6
    d['ET0_mm_h']   = (d['ET0_mm_day'] / 24).round(5)

    # Kc Bell Pepper — FAO-56 Table 11
    age = d['plant_age_days']
    d['Kc'] = np.where(
        age <= 25,  0.60,
        np.where(age <= 70,  0.60 + (1.05 - 0.60) * (age - 25) / 45,
        np.where(age <= 125, 1.05,
                 np.clip(1.05 + (0.90 - 1.05) * (age - 125) / 15, 0.90, 1.05)))
    ).round(4)
    d['ETc_mm_day']     = (d['Kc'] * d['ET0_mm_day']).round(4)        # Eq.58
    d['ETc_mm_h']       = (d['ETc_mm_day'] / 24).round(5)

    # Ks stress nước — FAO-56 Eq.84
    Dr  = np.maximum(SOIL_FC_R - d['soil_real'], 0)
    d['Ks'] = np.where(
        Dr <= RAW_R, 1.0,
        np.maximum((TAW_R - Dr) / (TAW_R * (1 - P_DEPLETION)), 0.0)
    ).round(4)
    d['ETc_adj_mm_day'] = (d['Ks'] * d['ETc_mm_day']).round(4)        # Eq.80
    d['ETc_adj_mm_h']   = (d['ETc_adj_mm_day'] / 24).round(5)

    # Lượng mưa hiệu quả (FAO USDA)
    r = d['rain_1h_mm']
    d['Peff_mm'] = np.where(r <= 75, r * (125 - 0.6*r) / 125, 125 + 0.1*r).round(3)
    d['IR_mm_h'] = np.maximum(d['ETc_adj_mm_h'] - d['Peff_mm'], 0.0).round(5)

    # BƯỚC 5: Soil — Phân tích độ ẩm đất

    d['soil_deficit'] = np.maximum(
        SOIL_TARGET - soil_clean,
        0
    ).round(2)
    d['soil_to_raw']  = np.maximum(
        soil_clean - SOIL_RAW_S, 
        0
    ).round(2)
    # Drying rate từ soil_clean — cap tối đa DRYING_RATE_MAX để không sai số lớn
    raw_dr_1h = d['dry_delta_soil'].rolling(2, min_periods=1).mean()
    raw_dr_3h = d['dry_delta_soil'].rolling(6, min_periods=1).mean()
    raw_dr_6h = d['dry_delta_soil'].rolling(12, min_periods=1).mean()

    # Cap drying rate: không quá DRYING_RATE_MAX %/30min
    d['soil_dr_1h'] = raw_dr_1h.clip(-DRYING_RATE_MAX, DRYING_RATE_MAX).round(3)
    d['soil_dr_3h'] = raw_dr_3h.clip(-DRYING_RATE_MAX, DRYING_RATE_MAX).round(3)
    d['soil_dr_6h'] = raw_dr_6h.clip(-DRYING_RATE_MAX, DRYING_RATE_MAX).round(3)

    if 'pump_on' in d.columns:
        pump = d['pump_on'].fillna(0).values
        cnt = np.zeros(len(d)); c = 0
        for i in range(len(d)):
            c = 0 if pump[i]==1 else c+1
            cnt[i] = c
        d['hours_since_irr'] = (cnt*0.5).round(1)
    else:
        d['hours_since_irr'] = 0.0


    # BƯỚC 6: Dự đoán soil 2h/4h/6h — cảm biến 80% + API 20%

    
    # Chỉ lấy phần âm của drying rate (đất mất nước)
    # nếu dương → 0 (đất đang ổn hoặc vừa tưới)
    dry_for_pred_3h = np.minimum(d['soil_dr_3h'], 0)  # %/30min, luôn ≤ 0
    dry_for_pred_6h = np.minimum(d['soil_dr_6h'], 0)  # %/30min, luôn ≤ 0

    # ETc đóng góp vào mất ẩm đất (mm/h → % soil mất)
    # Tính qua thể tích vùng cảm biến (270 cm³)
    etcadj_rate = d['ETc_adj_mm_h'] / SENSOR_VOL_L * 0.10  # %/h
    
    h_arr  = d['hour'].values
    hour_f = np.where(
        (h_arr >= 6) & (h_arr <= 18),
        np.sin(np.pi * (h_arr - 6) / 12) * 0.4 + 0.8,
        0.5
    )
    
    # VPD factor: chỉ tăng khi VPD thực sự cao, không nhân quá mạnh
    vpd_f = np.clip(1 + d['vpd_kPa']*0.03, 1.0, 1.20)
    rain_f = np.clip(1.0 - d['rain_prob_1h']/300, 0.80, 1.00)
    d['soil_pred_2h'] = np.clip(
        soil_clean
        + dry_for_pred_3h * 4  * vpd_f * hour_f * rain_f  # drying 2h (4 steps × 30min)
        - etcadj_rate * 2      * vpd_f,                   # ETc 2h
        10, soil_clean                                    # không vượt soil hiện tại
    ).round(2)

    d['soil_pred_4h'] = np.clip(
        soil_clean
        + dry_for_pred_6h * 8  * vpd_f * hour_f * rain_f  # drying 4h (8 steps)
        - etcadj_rate * 4      * vpd_f,                   # ETc 4h
        10, soil_clean
    ).round(2)

    d['soil_pred_6h'] = np.clip(
        soil_clean
        + dry_for_pred_6h * 12 * vpd_f * hour_f * rain_f   # drying 6h (12 steps)
        - etcadj_rate * 6      * vpd_f,                    # ETc 6h
        10, soil_clean
    ).round(2)

    
    # Điều chỉnh API (20%): mưa dự báo → soil giảm chậm hơn
    if api_available and 'forecast_temp_3h_c' in d.columns:
        fc_T3 = d['forecast_temp_3h_c']; fc_T6 = d['forecast_temp_6h_c']
        fc_H3 = d['forecast_hum_3h_pct']; fc_H6 = d['forecast_hum_6h_pct']
        fc_S3 = d['forecast_solar_3h'];   fc_S6 = d['forecast_solar_6h']

        def _fc_ET0(fcT, fcH, fcS):
            fes  = 0.6108 * np.exp((17.27*fcT) / (fcT+237.3))
            fvpd = fes * (1 - fcH/100)
            fRn  = (1-0.23) * fcS * 0.0864 * 0.77
            fdk  = (4098*fes) / ((fcT+237.3)**2)
            return np.maximum(
                (0.408*fdk*fRn + d['gamma_kPaC']*(900/(fcT+273))*u2*fvpd)
                / (fdk + d['gamma_kPaC']*(1+0.34*u2)), 0
            )

        fc_et0_3h = _fc_ET0(fc_T3, fc_H3, fc_S3)
        fc_et0_6h = _fc_ET0(fc_T6, fc_H6, fc_S6)

        # Điều chỉnh từ API
        tc3 = np.maximum(fc_T3 - T, 0) * 0.025
        tc6 = np.maximum(fc_T6 - T, 0) * 0.030
        rc3 = (d['rain_prob_3h']/100) * d['rain_3h_mm'] * 0.40
        rc6 = (d['rain_prob_6h']/100) * d['rain_6h_mm'] * 0.35
        ec3 = d['Kc'] * fc_et0_3h / 24 * 4
        ec6 = d['Kc'] * fc_et0_6h / 24 * 6

        api_adj_2h = -tc3*0.5 + rc3 - ec3*0.3
        api_adj_4h = -tc3     + rc3 - ec3*0.6
        api_adj_6h = -tc6     + rc6 - ec6

        # Blend: 80% cảm biến + 20% API
        d['soil_pred_2h'] = np.clip(0.80*d['soil_pred_2h'] + 0.20*(d['soil_pred_2h']+api_adj_2h), 10, soil_clean).round(2)
        d['soil_pred_4h'] = np.clip(0.80*d['soil_pred_4h'] + 0.20*(d['soil_pred_4h']+api_adj_4h), 10, soil_clean).round(2)
        d['soil_pred_6h'] = np.clip(0.80*d['soil_pred_6h'] + 0.20*(d['soil_pred_6h']+api_adj_6h), 10, soil_clean).round(2)
        d['ET0_fc_3h']    = (fc_et0_3h * d['Kc']).round(4)
        d['ET0_fc_6h']    = (fc_et0_6h * d['Kc']).round(4)
    else:
        # Fallback: chỉ dùng cảm biến
        d['soil_pred_2h'] = np.clip(d['soil_pred_2h'], 10, soil_clean).round(2)
        d['soil_pred_4h'] = np.clip(d['soil_pred_4h'], 10, soil_clean).round(2)
        d['soil_pred_6h'] = np.clip(d['soil_pred_6h'], 10, soil_clean).round(2)
        d['ET0_fc_3h']    = d['ET0_mm_day']
        d['ET0_fc_6h']    = d['ET0_mm_day']

    d['soil_def_2h'] = np.maximum(SOIL_RAW_S - d['soil_pred_2h'], 0).round(2)
    d['soil_def_4h'] = np.maximum(SOIL_RAW_S - d['soil_pred_4h'], 0).round(2)
    d['soil_def_6h'] = np.maximum(SOIL_RAW_S - d['soil_pred_6h'], 0).round(2)

    # Hours to dry: dùng drying rate âm thực tế
    eff_rate = dry_for_pred_6h * vpd_f * 2  # %/h, luôn ≤ 0
    d['hours_to_dry'] = np.where(
        eff_rate < -0.01,
        np.clip((soil_clean - SOIL_RAW_S) / np.abs(eff_rate), 0, 24),
        24.0   # rate=0 → đất không khô thêm → 24h
    ).round(1)

    # BƯỚC 7: Forecast điều chỉnh IR

    p1 = d['rain_prob_1h']/100
    p3 = d['rain_prob_3h']/100
    p6 = d['rain_prob_6h']/100
    d['exp_rain_1h']    = (p1*d['rain_1h_mm']).round(3)
    d['exp_rain_3h']    = (p3*d['rain_3h_mm']).round(3)
    d['exp_rain_6h']    = (p6*d['rain_6h_mm']).round(3)
    d['forecast_adj']   = (0.20*(d['exp_rain_1h']+0.5*d['exp_rain_3h']+0.3*d['exp_rain_6h'])).round(4)
    d['IR_adj']         = np.maximum(d['IR_mm_h'] - d['forecast_adj'], 0.0).round(5)
    d['rain_trend_1_3'] = (p3-p1).round(3)
    d['rain_trend_3_6'] = (p6-p3).round(3)


    # BƯỚC 8: Rolling features (xu hướng 3h, 6h)
    for col, alias in [
        ('temp_sensor_c','temp'), ('humidity_sensor_pct','hum'),
        ('lux_bh1750','lux'), ('soil_sensor_pct','soil'), ('vpd_kPa','vpd')
    ]:
        d[f'{alias}_r3h'] = d[col].rolling(6,  min_periods=1).mean().round(3)
        d[f'{alias}_r6h'] = d[col].rolling(12, min_periods=1).mean().round(3)

    # soil rolling dùng soil_clean
    d['soil_r3h'] = soil_clean.rolling(6,  min_periods=1).mean().round(3)
    d['soil_r6h'] = soil_clean.rolling(12, min_periods=1).mean().round(3)

    d['ET0_r6h']  = d['ET0_mm_h'].rolling(12, min_periods=1).mean().round(5)
    d['ET0_r12h'] = d['ET0_mm_h'].rolling(24, min_periods=1).mean().round(5)
    
    # BƯỚC 9: Encoding thời gian
    '''
    d['hour_sin']  = np.sin(2*np.pi*d['hour']/24).round(4)
    d['hour_cos']  = np.cos(2*np.pi*d['hour']/24).round(4)
    d['doy_sin']   = np.sin(2*np.pi*d['doy']/365).round(4)
    d['doy_cos']   = np.cos(2*np.pi*d['doy']/365).round(4)
    d['month_sin'] = np.sin(2*np.pi*d['month']/12).round(4)
    d['month_cos'] = np.cos(2*np.pi*d['month']/12).round(4)
    '''
    h = d['hour']
    d['time_phase']    = np.where(h<5,0,np.where(h<8,1,np.where(h<11,2,
                          np.where(h<14,3,np.where(h<18,4,5)))))
    d['is_dry_season'] = ((d['month']<=4)|(d['month']>=12)).astype(int)

    return d


# Danh sách features dùng để train/predict
# =========================
# Nhóm 1: Thông tin cơ bản
# =========================
FEATURE_BASIC = [
    "hour",
    "plant_age_days",
    "soil_clean",
    "soil_deficit",
    
    "soil_r3h",
    "soil_r6h",
    "delta_soil",
]

# =========================
# Nhóm 2: Vi khí hậu
# =========================
FEATURE_ENV = [
    "temp_sensor_c",
    "humidity_sensor_pct",

    "solar_radiation_wm2",
    "wind_speed_ms",

]

# =========================
# Nhóm 3: Đặc trưng FAO-56
# =========================
FEATURE_FAO = [
    "ET0_mm_day",
    "vpd_kPa",
    "Ks",
]

# =========================
# Nhóm 4: Đặc trưng dự báo
# =========================
FEATURE_FORECAST = [
    "hours_to_dry",
    "soil_pred_6h",
]

# =========================
# Nhóm 5: Luật tưới
# =========================
FEATURE_RULE = [
    "is_hot_sun",
    "is_good_hour",
    "is_morning",
    "is_evening",
]
FEATURE_NGON = [
    "hour",
    "soil_clean",
    "temp_sensor_c",
    "humidity_sensor_pct",
    "ET0_mm_day",
    "Ks",
    "is_good_hour",
    "is_morning",
    "is_evening",
    "soil_future_pred"
]
FEATURES_SOIL = [
    "hour",
    "plant_age_days",
    "soil_clean",
    "soil_deficit",

    "delta_soil",

    "temp_sensor_c",
    "humidity_sensor_pct",

    "solar_radiation_wm2",
    "wind_speed_ms",

    "ET0_mm_day",
    "vpd_kPa",
    "Ks",

    "is_hot_sun",
    "is_good_hour",
    "is_morning",
    "is_evening",
]
# =========================
# Full feature
# =========================
FEATURES = (
    FEATURE_BASIC
    + FEATURE_ENV
    + FEATURE_FAO
    + FEATURE_FORECAST
    + FEATURE_RULE
)
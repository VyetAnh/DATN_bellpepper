"""
predictor.py — Dự đoán quyết định tưới.
"""

import logging
import numpy as np
import pandas as pd
import joblib

from config import *
from features import (
    compute_features,
    FEATURES,
    FEATURES_SOIL,
    FEATURE_NGON
)
from labels import compute_fao56_labels
from leaf_analyzer import get_adjust_percent
logger = logging.getLogger(__name__)

_history: list[dict] = []


def load_models():
    reg_soil = joblib.load(f'{MODEL_DIR}/rf_regressor_soil.pkl')
    reg_vol = joblib.load(f'{MODEL_DIR}/rf_regressor_volume.pkl')
    logger.info("✅ Models loaded")
    return reg_soil, reg_vol


def predict(sensor: dict, reg_soil, reg_vol) -> dict:
    """
    Dự đoán quyết định tưới.
    Điều chỉnh volume theo điều kiện thực tế:
    - Soil đủ   → giảm volume hoặc tắt
    - Nắng đỉnh (11h-14h) hoặc tối khuya (22h-4h) → CẤM (hại cây rõ ràng)
    """
    global _history

    row = dict(sensor)
    if 'timestamp' not in row:
        row['timestamp'] = pd.Timestamp.now()
    row.setdefault('pump_on', 0)

    _history.append(row)
    if len(_history) > 72:
        _history.pop(0)

    df_h = pd.DataFrame(_history)
    df_f = compute_features(df_h, api_available=True)
    df_lbl = compute_fao56_labels(df_f)
    
    r = df_f.iloc[-1]
    lbl = df_lbl.iloc[-1]
    
    soil_now  = float(r['soil_clean'])
    soil_p6h  = float(r['soil_pred_6h'])
    hour_now  = int(r['hour'])
    leaf_adjust = get_adjust_percent()
    logger.info(
        f"🍃 Leaf adjust: {leaf_adjust}%"
    )

    X_soil = df_f[FEATURES_SOIL].fillna(0).iloc[[-1]]
    
    if reg_soil is not None:
        soil_future_pred = float(
            np.clip(
                reg_soil.predict(X_soil)[0],
                0,
                100
            )
        )
    else:
        soil_future_pred = float(
            r['soil_pred_6h']
        )
    logger.info(
        f"🌱 RF Soil: {soil_now:.1f}% → {soil_future_pred:.1f}%"
    )
    logger.info(
        f"🌱 FAO Soil 6h: {soil_p6h:.1f}%"
    )
    X_vol = df_f.iloc[[-1]].copy()

    X_vol["soil_future_pred"] = soil_future_pred

    X_vol = X_vol.reindex(
        columns=reg_vol.feature_names_in_,
        fill_value=0
    )

    X_vol["soil_future_pred"] = soil_future_pred
    override_reason = None
    pump_on = int(lbl['pump_on'])
    if soil_now < SOIL_RAW_S:
        pump_on = 1
    elif (
        soil_now < SOIL_WARNING
        and soil_future_pred < SOIL_RAW_S
    ):
        pump_on = 1
    if pump_on:
        try:
            vol_L = float(
                np.maximum(
                    reg_vol.predict(X_vol)[0],
                    0
                )
            )
        except Exception as e:
            logger.warning(f"RF volume lỗi: {e}")
            vol_L = float(lbl['irrigation_volume_L'])
        vol_L = max(vol_L, 0.01)

        if leaf_adjust != 0:

            vol_L *= (
                1 + leaf_adjust / 100
            )

            override_reason = (
                f"Leaf adjust {leaf_adjust:+d}%"
            )

        dur_s = vol_L / PUMP_FLOW_LPS
    else:
        vol_L = 0.0
        dur_s = 0.0

    '''
    rain_now  = float(sensor.get('rain_1h_mm', 0))
    rain_prob = float(sensor.get('rain_prob_1h', 0))
    rain_3h   = float(sensor.get('rain_3h_mm', 0))
    '''

#    volume_adj_reasons = []

    # CẤM TUYỆT ĐỐI (chỉ 2 trường hợp gây hại rõ ràng) 

    # 1. Nắng đỉnh 11h–14h: tưới lúc này gây sốc nhiệt rễ, cháy lá
    if hour_now in [12, 13, 14]:

        if soil_now >= SOIL_RAW_S:
            pump_on = 0
            dur_s = 0
            vol_L = 0

            override_reason = (
                f"Cấm tưới {hour_now}h"
            )

    # 2. Tối khuya 22h–4h: lá ướt qua đêm dài → nấm bệnh
    elif hour_now in list(range(22,24)) + list(range(0,5)):
        if soil_now < SOIL_RAW_S:
            pump_on = 1
            vol_L = min(max(vol_L, 0.02), 0.03)
            dur_s = vol_L / PUMP_FLOW_LPS
            override_reason = (
                f"Tưới đêm duy trì ({hour_now}h)"
            )
        else:
            pump_on = 0
            dur_s = 0.0
            vol_L = 0.0


    elif pump_on == 1 and vol_L > 0:
        
        # 3. Soil đang đủ ẩm (≥80%) — giảm 80% volume (tưới rất ít)
        if soil_future_pred >= SOIL_OPTIMAL_HI:
            pump_on = 0
            vol_L = 0
            dur_s = 0
        else:
            vol_L = max(vol_L, 0.010)
            dur_s = vol_L / PUMP_FLOW_LPS
        '''
        # 4. Soil tốt (70–84%) và pred 4h vẫn ổn — giảm 50%
        elif soil_now >= SOIL_OPTIMAL_LO and soil_p4h >= SOIL_WARNING:
            vol_L *= 0.50
            volume_adj_reasons.append(f"soil={soil_now:.0f}% ổn, pred4h={soil_p4h:.0f}% → ×0.50")
        
        # 5. Đang mưa to (≥5mm) — giảm 70%
        if rain_now >= 5.0:
            vol_L *= 0.30
            volume_adj_reasons.append(f"mưa to {rain_now}mm → ×0.30")

        # 6. Đang mưa vừa (2–5mm) — giảm 50%
        elif rain_now >= 2.0:
            vol_L *= 0.50
            volume_adj_reasons.append(f"mưa vừa {rain_now}mm → ×0.50")

        # 7. Mưa nhỏ (0.5–2mm) — giảm 20%
        elif rain_now >= 0.5:
            vol_L *= 0.80
            volume_adj_reasons.append(f"mưa nhỏ {rain_now}mm → ×0.80")

        # 8. Sắp mưa rất to (prob ≥ 80%) — giảm 60%
        if rain_prob >= 80:
            vol_L *= 0.40
            volume_adj_reasons.append(f"sắp mưa to {rain_prob}% → ×0.40")

        # 9. Sắp mưa to (prob 60–79%) — giảm 35%
        elif rain_prob >= 60:
            vol_L *= 0.65
            volume_adj_reasons.append(f"sắp mưa {rain_prob}% → ×0.65")

        # 10. Sắp mưa nhẹ (prob 30–59%) — giảm 15%
        elif rain_prob >= 30:
            vol_L *= 0.85
            volume_adj_reasons.append(f"có thể mưa {rain_prob}% → ×0.85")
        

        if volume_adj_reasons:
            override_reason = "Giảm volume: " + " | ".join(volume_adj_reasons)
        '''
    if override_reason:
        logger.info(f"⚡ {override_reason}")
    is_hot_sun = int(hour_now in [12, 13, 14])

    is_good_hour = int(
        (hour_now in MORNING_HOURS)
        or
        (hour_now in EVENING_HOURS)
    )

    return {
        'pump_on':         pump_on,
        'xac_suat': 1.0 if pump_on else 0.0,
        'relay_on_ms':     int(dur_s * 1000),
        'duration_s':      round(dur_s, 2),
        'volume_mL':       round(vol_L * 1000, 1),
        'volume_L':        round(vol_L, 5),
        'override_reason': override_reason or "",
        # Soil
        'soil_now_pct':    round(soil_now, 1),
        'soil_future_pred':round(soil_future_pred, 1),
        'soil_pred_6h':    round(soil_p6h, 1),
        'hours_to_dry':    round(float(r['hours_to_dry']), 1),
        # FAO-56
        'ET0_mm_day':      round(float(r['ET0_mm_day']), 3),
        'ETc_adj':         round(float(r['ETc_adj_mm_day']), 3),
        'Kc':              round(float(r['Kc']), 3),
        'Ks':              round(float(r['Ks']), 3),
        'vpd_kPa':         round(float(r['vpd_kPa']), 3),
        
        'leaf_adjust': leaf_adjust,
        'is_hot_sun':      is_hot_sun,
        'is_good_hour':    is_good_hour,
    }


def reset_history():
    global _history
    _history = []

# labels.py — Tính nhãn FAO-56 + logic tưới theo độ ẩm đất.

import numpy as np
import pandas as pd
from config import *

def compute_fao56_labels(df_feat: pd.DataFrame) -> pd.DataFrame:
    d    = df_feat.copy()
    soil = d['soil_clean']
    hour = d['hour']

    # Điều kiện CẤM tuyệt đối 
    #bad_hour   = hour.isin(BAD_HOURS)
    '''
    raining    = d['rain_1h_mm'] >= 2.0      # mưa lớn thực tế
    rain_soon  = d['rain_prob_1h'] >= 95.0   # rất có thể mưa
    '''
    soil_full  = soil >= SOIL_OPTIMAL_HI     # đủ rồi
   
    morning_hour = hour.isin(MORNING_HOURS)
    evening_hour = hour.isin(EVENING_HOURS)

    good_hour = morning_hour | evening_hour

    # Phân tầng theo độ ẩm đất
    # Tầng 1: < 57% — KHẨN CẤP
    urgent   = soil < SOIL_RAW_S                                    
    # Tầng 2: 57–59% — RAW threshold
    at_raw   = (soil >= SOIL_RAW_S)   & (soil < SOIL_WARNING)        
    # Tầng 3: 60–64% — hơi khô
    warn_dry = (soil >= SOIL_RAW_R) & (soil < SOIL_OPTIMAL_LO)      
    # Tầng 4: 65–80% — dưới vùng ổn định, cần chủ động tưới trước
    below_ok = (soil >= SOIL_OPTIMAL_LO) & (soil < SOIL_OPTIMAL_HI)   
    evening_rescue = evening_hour & (soil < SOIL_EVENING_RESCUE)

    need_4h = d['soil_pred_4h'] < SOIL_OPTIMAL_LO
    need_6h = d['soil_pred_6h'] < SOIL_OPTIMAL_LO


    d['pump_on'] = (
        ~soil_full
        & (

            # <57% : tưới mọi lúc, trừ 12-15h
            (urgent & ~hour.isin([12, 13, 14]))

            # 60-70% : chỉ tưới sáng hoặc tối
            | (at_raw & morning_hour)

            # 60-65% : tối chỉ tưới cứu chiều
            | (warn_dry & evening_rescue)

            # 65-80% : chỉ tưới chủ động buổi sáng
            | (below_ok & morning_hour & (need_4h | need_6h))
        )
    ).astype(int)

    # ── Tính thể tích nước 

    # Mức target thực tế (không vượt 80% để không úng nước)
    effective_target = min(SOIL_TARGET, 80.0)
    target_soil = np.where(
        evening_hour,
        SOIL_TARGET_EVENING,
        effective_target
    )

    # Độ ẩm tham chiếu (80% hiện tại + 20% dự báo)
    soil_ref = 0.8 * soil + 0.2 * d['soil_pred_6h']

    # Volume cơ bản: bù thiếu hụt đến target trong 6h tới
    vol_base = (
        np.maximum(target_soil - soil_ref, 0)
        / 100
        * WATER_CALC_VOL_L
        * COIR_WATER_RETENTION
    )

    # ETc đóng góp nhỏ (5% thay vì chính)
    vol_etc = d['ETc_adj_mm_h'] * 6 * WATER_CALC_VOL_L / 100 * 0.05

    # VPD bonus nếu rất khô nóng
#    vol_vpd = np.where(d['vpd_kPa'] > 2.5, WATER_CALC_VOL_L * COIR_WATER_RETENTION * 0.01, 0)

    # Khẩn cấp (<57%)
    vol_urgent = np.where(
        soil < SOIL_RAW_S,
        (
            np.maximum(effective_target - soil, 0)
            / 100
            * WATER_CALC_VOL_L
            * COIR_WATER_RETENTION
            * 0.10
        ),
        0
    )

    # Tổng
    vol_total = vol_base + vol_etc + vol_urgent

    # Giới hạn an toàn
    vol_max = np.maximum(80.0 - soil, 0) / 100 * WATER_CALC_VOL_L * COIR_WATER_RETENTION
    vol_min = 0.010   

    vol = np.where(
        d['pump_on'] == 1,
        np.clip(vol_total, vol_min, vol_max),
        0.0
    )

    d['irrigation_volume_L']   = vol.round(4)
    d['irrigation_duration_s'] = np.where(
        d['pump_on'] == 1,
        (vol / PUMP_FLOW_LPS).round(2),
        0.0
    )
    return d

"""
trainer.py — Train RF models 
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from config import *
from features import (
    compute_features,
    FEATURES,
    FEATURES_SOIL,
    FEATURE_BASIC,
    FEATURE_ENV,
    FEATURE_FAO,
    FEATURE_NGON,
    FEATURE_FORECAST,
    FEATURE_RULE
)
from labels import compute_fao56_labels
###so sánh các model với nhau
from sklearn.svm import SVR
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

GRAPH_DIR = "FullDataset/graph/rf"

Path(GRAPH_DIR).mkdir(
    parents=True,
    exist_ok=True
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
###
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)


def load_and_prepare_data() -> pd.DataFrame | None:

    if not os.path.exists(CSV_RAW):
        logger.error(f"❌ Không tìm thấy {CSV_RAW}")
        return None

    df_raw = pd.read_csv(CSV_RAW, parse_dates=['timestamp'])

    # TIME FEATURES
    dt = pd.to_datetime(df_raw["timestamp"])

    # chỉ lấy thời gian trong ngày
    df_raw["hour"] = dt.dt.hour

    # hoặc dùng dạng liên tục
    df_raw["time_float"] = (
        dt.dt.hour +
        dt.dt.minute / 60.0
    )
    logger.info(f"📊 Đọc CSV: {len(df_raw):,} hàng × {df_raw.shape[1]} cột")

    if len(df_raw) < 100:
        logger.warning(f"⚠️  Chưa đủ dữ liệu ({len(df_raw)} hàng, cần ≥ 100)")
        return None

    # Tính features
    if 'pump_on' in df_raw.columns:
        df_tmp = df_raw.copy()
        df_tmp['pump_on'] = df_raw['pump_on'].fillna(0)
        df_feat = compute_features(df_tmp)
    else:
        df_feat = compute_features(df_raw)

    # Tính nhãn FAO-56
    df_fao = compute_fao56_labels(df_feat)

    # Blend nhãn
    if 'pump_on' in df_raw.columns:
        df_feat['pump_on'] = df_raw['pump_on'].fillna(0).astype(int)

        real_vol = df_raw['volume_mL'].fillna(0) / 1000

        real_vol = df_raw['volume_mL'].fillna(0) / 1000  # mL -> L

        df_feat['irrigation_duration_s'] = np.where(
            df_raw['pump_on'] == 1,
            real_vol / PUMP_FLOW_LPS,
            df_fao['irrigation_duration_s']
        )
        fao_vol = df_fao['irrigation_volume_L']
        mix_vol = 0.9 * real_vol + 0.1 * fao_vol
        df_feat['irrigation_volume_L'] = np.where(
            df_raw['pump_on'] == 1,
            mix_vol,
            fao_vol
        )
        
        
        # Soil sau 3 giờ (36 mẫu × 5 phút)
        future_pump = pd.Series(
            [
                df_raw['pump_on'].iloc[i+1:i+37].max()
                if i + 36 < len(df_raw)
                else np.nan
                for i in range(len(df_raw))
            ]
        )

        # Nhãn cho RF Soil
        df_feat['soil_future'] = (
            df_feat['soil_clean']
            .shift(-36)
        )
        # 
        df_feat['soil_future'] += np.random.normal(
            0,
            1.2,
            len(df_feat)
        )

        df_feat['soil_future'] = (
            df_feat['soil_future']
            .clip(0, 100)
        )
        # Chỉ dùng cho RF Soil
        df_feat['soil_valid'] = (
            (future_pump == 0)
            & df_feat['soil_future'].notna()
        )
        
    else:
        df_feat['pump_on'] = df_fao['pump_on']
        df_feat['irrigation_duration_s'] = df_fao['irrigation_duration_s']
        df_feat['irrigation_volume_L'] = df_fao['irrigation_volume_L']
    return df_feat


def train_models(df: pd.DataFrame) -> dict:
    """
    Train RF models và lưu vào MODEL_DIR.
    """
    Path(MODEL_DIR).mkdir(
        parents=True,
        exist_ok=True
    )
    
    df_volume = df[
        FEATURES +
        ['pump_on', 'irrigation_volume_L']
    ].dropna()

    df_soil = df[
        FEATURES +
        ['soil_future', 'soil_valid']
    ].dropna()

    df_soil = df_soil[
        df_soil['soil_valid']
    ].copy() 
   
    logger.info(f"🤖 Train với {len(df_volume):,} mẫu | Features: {len(FEATURES)}")

    # ===== RF Volume =====
    X = df_volume[FEATURES]
    y_vol = df_volume['irrigation_volume_L']

    # ===== RF Soil =====
    X_soil = df_soil[FEATURES_SOIL]
    y_soil = df_soil['soil_future']

    # RF SOIL

    X_soil_train, X_soil_test, y_soil_tr, y_soil_te = train_test_split(
        X_soil,
        y_soil,
        test_size=0.20,
        random_state=42
    )

    logger.info(
        f"🌱 Soil samples train={len(X_soil_train)} "
        f"test={len(X_soil_test)}"
    )

    # RF VOLUME

    df_irr = df_volume[
        df_volume["pump_on"] == 1
    ].copy()

    X_vol = df_irr.drop(
        columns=[
            "irrigation_volume_L"
        ]
    )

    y_vol = df_irr["irrigation_volume_L"]
    '''
    split_vol = int(len(df_irr) * 0.80)

    X_train = X_vol.iloc[:split_vol]
    X_test = X_vol.iloc[split_vol:]

    y_vol_tr = y_vol.iloc[:split_vol]
    y_vol_te = y_vol.iloc[split_vol:]
    '''
    X_train, X_test, y_vol_tr, y_vol_te = train_test_split(
        X_vol,
        y_vol,
        test_size=0.20,
        random_state=42
    )
    X_train_soil_pred = X_train[FEATURES_SOIL]
    X_test_soil_pred  = X_test[FEATURES_SOIL]

    X_train_vol = X_train[FEATURE_NGON[:-1]].copy()
    X_test_vol  = X_test[FEATURE_NGON[:-1]].copy()
    logger.info(
        f"💧 Volume samples train={len(X_train)} "
        f"test={len(X_test)}"
    )

    # Ablation Feature Sets


    feature_sets = {

        "Basic":
            FEATURE_BASIC,

        "+Env":
            FEATURE_BASIC +
            FEATURE_ENV,

        "+FAO":
            FEATURE_BASIC +
            FEATURE_ENV +
            FEATURE_FAO,

        "+Forecast":
            FEATURES,

        "Full":
            FEATURE_NGON,
    }

    ablation_results = []
    results = {}

    if len(X_soil_train) < 5:
        raise ValueError(
            f"Không đủ dữ liệu train soil ({len(X_soil_train)} mẫu)"
        )
    n_splits = min(5, len(X_soil_train))
    if n_splits < 2:
        raise ValueError("Không đủ dữ liệu để train")

    kf = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    

    # Model 1: RF Regressor (soil_future)
    logger.info("[1/4] Training RF Regressor — soil_future...")

    reg_soil = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features='sqrt',
        n_jobs=-1,
        random_state=RF_RANDOM_STATE
    )

    cv_mae_s = -cross_val_score(
        reg_soil,
        X_soil_train,
        y_soil_tr,
        cv=kf,
        scoring='neg_mean_absolute_error',
        n_jobs=-1
    )
    print("\nSOIL FEATURES")
    print(X_soil.columns.tolist())
    reg_soil.fit(X_soil_train, y_soil_tr)
    soil_future_train = reg_soil.predict(
        X_train_soil_pred
    )

    soil_future_test = reg_soil.predict(
        X_test_soil_pred
    )

    X_train_soil_pred = X_train[FEATURES_SOIL]
    X_test_soil_pred  = X_test[FEATURES_SOIL]
    X_train_vol = X_train.copy()
    X_test_vol  = X_test.copy()

    X_train_vol['soil_future_pred'] = soil_future_train
    X_test_vol['soil_future_pred'] = soil_future_test
    soil_future_all = np.concatenate([
        soil_future_train,
        soil_future_test
    ])
    X_irr = X_train_vol
    X_irr_test = X_test_vol
    logger.info(
        f"RF main features = {len(X_irr.columns)}"
    )

    logger.info(
        f"{list(X_irr.columns)}"
    )
    logger.info(
        f"💧 Volume samples train={len(X_irr)} "
        f"test={len(X_irr_test)}"
    )
    logger.info(
        f"💧 Mẫu tưới dùng train volume: {len(X_irr):,}"
    )
    print("\n===== VOLUME STATS =====")

    print(
        df_irr["irrigation_volume_L"]
        .describe()
    )

    print("\n===== TOP VOLUME =====")

    print(
        df_irr["irrigation_volume_L"]
        .round(3)
        .value_counts()
        .sort_index()
    )
    print(
        y_vol_te.min(),
        y_vol_te.max(),
        y_vol_te.std()
    )
    if len(X_irr) < 10:
        raise ValueError(
            f"Quá ít mẫu tưới để train volume model ({len(X_irr)} mẫu)"
        )
    yp_soil = reg_soil.predict(X_soil_test)

    mae_s = mean_absolute_error(
        y_soil_te,
        yp_soil
    )

    rmse_s = np.sqrt(
        mean_squared_error(
            y_soil_te,
            yp_soil
        )
    )

    r2_s = r2_score(
        y_soil_te,
        yp_soil
    )
    # ==========================
    # Soil Actual vs Predicted
    # ==========================

    plt.figure(figsize=(6,6))

    plt.scatter(
        y_soil_te,
        yp_soil,
        alpha=0.6
    )

    mn = min(
        y_soil_te.min(),
        yp_soil.min()
    )

    mx = max(
        y_soil_te.max(),
        yp_soil.max()
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        'r--'
    )

    plt.xlabel("Actual Soil (%)")
    plt.ylabel("Predicted Soil (%)")
    plt.title("RF Soil Prediction")

    plt.tight_layout()

    plt.savefig(
        f"{GRAPH_DIR}/soil_scatter.png",
        dpi=300
    )

    plt.close()
    joblib.dump(
        reg_soil,
        f'{MODEL_DIR}/rf_regressor_soil.pkl'
    )

    logger.info(
        f"   Soil MAE: {mae_s:.3f}% | "
        f"RMSE: {rmse_s:.3f}% | "
        f"R²: {r2_s:.4f}"
    )
    results['soil_mse'] = float(
        mean_squared_error(
            y_soil_te,
            yp_soil
        )
    )
    results['soil_mae'] = float(mae_s)
    results['soil_rmse'] = float(rmse_s)
    results['soil_r2'] = float(r2_s)

    # Model 2: RF Regressor (volume_L) 
    logger.info("[2/4] Training RF Regressor — volume_L...")
    y_vol2 = y_vol_tr
    reg_vol = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS, min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_features='sqrt', n_jobs=-1, random_state=RF_RANDOM_STATE
    )
    if len(X_irr) < 5:
        raise ValueError(
            f"Không đủ mẫu tưới cho KFold ({len(X_irr)} mẫu)"
        )
        
        
    kf_vol = KFold(
        n_splits=min(5, len(X_irr)),
        shuffle=True,
        random_state=RF_RANDOM_STATE
    )
    cv_mae_v = -cross_val_score(reg_vol, X_irr, y_vol2, cv=kf_vol,
                                 scoring='neg_mean_absolute_error', n_jobs=-1)

    print("===== RF VOLUME TRAIN FEATURES =====")
    print(X_irr.columns.tolist())

    reg_vol.fit(X_irr, y_vol2)
    import shap

    explainer = shap.TreeExplainer(reg_vol)

    shap_values = explainer.shap_values(X_irr)

    shap.summary_plot(
        shap_values,
        X_irr,
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        f"{GRAPH_DIR}/shap_summary.png",
        dpi=300
    )

    plt.close()
    if len(X_irr_test) > 0:
        yp_vol = np.maximum(
            reg_vol.predict(
                X_test_vol
            ),
            0
        )

        mae_v = mean_absolute_error(
            y_vol_te,
            yp_vol
        )

        rmse_v = np.sqrt(
            mean_squared_error(
                y_vol_te,
                yp_vol
            )
        )

        r2_v = r2_score(
            y_vol_te,
            yp_vol
        )
        # Feature Importance

        importance = pd.DataFrame({

            "Feature":
                X_irr.columns,

            "Importance":
                reg_vol.feature_importances_
        })

        importance = importance.sort_values(
            "Importance",
            ascending=True
        )

        importance.to_csv(
            f"{GRAPH_DIR}/feature_importance.csv",
            index=False
        )

        plt.figure(figsize=(7,5))

        plt.barh(
            importance["Feature"],
            importance["Importance"]
        )

        plt.xlabel("Importance")
        plt.tight_layout()

        plt.savefig(
            f"{GRAPH_DIR}/feature_importance.png",
            dpi=300
        )

        plt.close()
        # Volume Actual vs Predicted

        plt.figure(figsize=(6,6))

        plt.scatter(
            y_vol_te,
            yp_vol,
            alpha=0.6
        )

        mn = min(
            y_vol_te.min(),
            yp_vol.min()
        )

        mx = max(
            y_vol_te.max(),
            yp_vol.max()
        )

        plt.plot(
            [mn, mx],
            [mn, mx],
            'r--'
        )

        plt.xlabel("Actual Volume (L)")
        plt.ylabel("Predicted Volume (L)")
        plt.title("RF Volume Prediction")

        plt.tight_layout()

        plt.savefig(
            f"{GRAPH_DIR}/volume_scatter.png",
            dpi=300
        )

        plt.close()
        mse_v = mean_squared_error(
            y_vol_te,
            yp_vol
        )

    else:
        mse_v = 0.0
        mae_v = 0.0
        rmse_v = 0.0
        r2_v = 0.0
    joblib.dump(reg_vol, f'{MODEL_DIR}/rf_regressor_volume.pkl')
    for name, feats in feature_sets.items():

        feats_real = [
            f for f in feats
            if f != "soil_future_pred"
        ]

        X_ab = df_irr[feats_real].copy()

        if "soil_future_pred" in feats:
            X_ab["soil_future_pred"] = soil_future_all



        if name == "Full":

            logger.info(
                f"ABLATION FULL FEATURES = {len(X_ab.columns)}"
            )

            logger.info(
                f"{list(X_ab.columns)}"
            )


        X_train_ab, X_test_ab, y_train_ab, y_test_ab = train_test_split(
            X_ab,
            y_vol,
            test_size=0.20,
            random_state=42
        )


        rf = RandomForestRegressor(
            n_estimators=RF_N_ESTIMATORS,
            min_samples_leaf=RF_MIN_SAMPLES_LEAF,
            max_features='sqrt',
            random_state=RF_RANDOM_STATE,
            n_jobs=-1
        )

        rf.fit(
            X_train_ab,
            y_train_ab
        )
        pred = rf.predict(
            X_test_ab
        )

        r2 = r2_score(
            y_test_ab,
            pred
        )

        mae = mean_absolute_error(
            y_test_ab,
            pred
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test_ab,
                pred
            )
        )

        ablation_results.append({
            "Features": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        logger.info(
            f"{name}: "
            f"MAE={mae:.5f} "
            f"RMSE={rmse:.5f} "
            f"R²={r2:.4f}"
        )
    logger.info(f"   CV MAE: {cv_mae_v.mean():.5f}L | Test MAE: {mae_v:.5f}L R²: {r2_v:.4f}")
    results['volume_mse'] = float(mse_v)
    results['volume_mae'] = float(mae_v)
    results['volume_rmse'] = float(rmse_v)
    results['volume_r2']  = float(r2_v)
    # train xgboost
    logger.info("[3/4] Training XGBoost...")

    xgb = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42
    )

    xgb.fit(X_irr, y_vol2)

    yp_xgb = np.maximum(
        xgb.predict(X_irr_test),
        0
    )

    mse_xgb = mean_squared_error(
        y_vol_te,
        yp_xgb
    )

    mae_xgb = mean_absolute_error(
        y_vol_te,
        yp_xgb
    )

    rmse_xgb = np.sqrt(mse_xgb)

    r2_xgb = r2_score(
        y_vol_te,
        yp_xgb
    )
    joblib.dump(
        xgb,
        f"{MODEL_DIR}/xgb_volume.pkl"
    )
    # train SVR
    logger.info("[4/4] Training SVR...")

    svr = SVR(
        kernel="rbf",
        C=10
    )

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    svr = make_pipeline(
        StandardScaler(),
        SVR(
            kernel="rbf",
            C=10
        )
    )
    svr.fit(
        X_irr,
        y_vol2
    )
    yp_svr = svr.predict(
        X_irr_test
    )

    mse_svr = mean_squared_error(
        y_vol_te,
        yp_svr
    )

    mae_svr = mean_absolute_error(
        y_vol_te,
        yp_svr
    )

    rmse_svr = np.sqrt(mse_svr)

    r2_svr = r2_score(
        y_vol_te,
        yp_svr
    )
    joblib.dump(
        svr,
        f"{MODEL_DIR}/svr_volume.pkl"
    )
    ## DATAFrame
    compare = pd.DataFrame({

        "Model": [
            "RandomForest",
            "XGBoost",
            "SVR"
        ],

        "MSE": [
            mse_v,
            mse_xgb,
            mse_svr
        ],

        "MAE": [
            mae_v,
            mae_xgb,
            mae_svr
        ],

        "RMSE": [
            rmse_v,
            rmse_xgb,
            rmse_svr
        ],

        "R2": [
            r2_v,
            r2_xgb,
            r2_svr
        ]
    })
    ## lưu bảng
    compare.to_csv(
        f"{GRAPH_DIR}/compare_metrics.csv",
        index=False
    )

    compare.to_string(
        open(
            f"{GRAPH_DIR}/compare_metrics.txt",
            "w",
            encoding="utf8"
        )
    )
    ## vẽ MSE, MAE, RMSE, R2
    plt.figure(figsize=(6,4))
    plt.bar(
        compare["Model"],
        compare["MSE"]
    )
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(
        f"{GRAPH_DIR}/mse_compare.png",
        dpi=300
    )
    plt.close()
    plt.figure(figsize=(6,4))
    plt.bar(
        compare["Model"],
        compare["MAE"]
    )
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(
        f"{GRAPH_DIR}/mae_compare.png",
        dpi=300
    )
    plt.close()
    plt.figure(figsize=(6,4))
    plt.bar(
        compare["Model"],
        compare["RMSE"]
    )
    plt.ylabel("RMSE")
    plt.tight_layout()
    plt.savefig(
        f"{GRAPH_DIR}/rmse_compare.png",
        dpi=300
    )
    plt.close()
    plt.figure(figsize=(6,4))
    plt.bar(
        compare["Model"],
        compare["R2"]
    )
    plt.ylabel("R²")
    plt.tight_layout()
    plt.savefig(
        f"{GRAPH_DIR}/r2_compare.png",
        dpi=300
    )
    plt.close()
    ###
    
    # Lưu thông tin lần train cuối
    ablation = pd.DataFrame(
        ablation_results
    )

    ablation.to_csv(
        f"{GRAPH_DIR}/rf_ablation.csv",
        index=False
    )

    plt.figure(figsize=(6,4))

    plt.bar(
        ablation["Features"],
        ablation["R2"]
    )

    plt.ylabel("R²")

    plt.tight_layout()

    plt.savefig(
        f"{GRAPH_DIR}/rf_ablation.png",
        dpi=300
    )

    plt.close()
    results['trained_at']  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results['n_samples']   = len(df_volume)

    import json
    with open(f'{MODEL_DIR}/train_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"✅ Train xong! Models lưu tại: {MODEL_DIR}/")
    logger.info(
        f"Volume MAE={results['volume_mae']}L | "
        f"RMSE={results['volume_rmse']}L | "
        f"R²={results['volume_r2']}"
    )
    return results


def train_once():
    """Chạy 1 lần train đầy đủ."""
    logger.info("=" * 55)
    logger.info(f"🚀 Bắt đầu train: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 55)

    df = load_and_prepare_data()
    if df is None:
        return False

    results = train_models(df)
    return results
if __name__ == "__main__":
    import traceback

    try:
        result = train_once()

        if result:
            print("\n===== TRAIN THÀNH CÔNG =====")
            print(result)
        else:
            print("\n===== KHÔNG TRAIN ĐƯỢC =====")

    except Exception:
        print("\n===== TRACEBACK =====")
        traceback.print_exc()
        

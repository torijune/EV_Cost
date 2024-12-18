import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib
import optuna

from preprocessor import main as DataPreprocessor

def objective(trial, X, y, k_folds=5):
    """
    Optuna 목적 함수: K-fold 교차 검증으로 모델의 하이퍼파라미터를 최적화합니다.
    """
    model_name = trial.suggest_categorical("model", ["RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost"])
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_rmse = []

    if model_name == "RandomForest":
        model = RandomForestRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            random_state=42
        )
    elif model_name == "ExtraTrees":
        model = ExtraTreesRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            random_state=42
        )
    elif model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            random_state=42
        )
    elif model_name == "LightGBM":
        model = LGBMRegressor(
            n_estimators=trial.suggest_int("n_estimators", 50, 200),
            max_depth=trial.suggest_int("max_depth", 5, 50),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            random_state=42,
            verbose=-1
        )
    elif model_name == "CatBoost":
        model = CatBoostRegressor(
            iterations=trial.suggest_int("iterations", 50, 200),
            depth=trial.suggest_int("depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3),
            random_state=42,
            verbose=0
        )

    # K-fold 교차 검증 수행
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        fold_rmse.append(rmse)

    return np.mean(fold_rmse)  # 평균 RMSE 반환

def train_with_kfold_and_save(data_path, target_column, drop_columns, model_save_path, n_trials=50, k_folds=5):
    """
    Optuna와 K-fold 교차 검증을 사용해 모델을 학습하고 최적의 모델을 저장합니다.

    Parameters:
        data_path (str): 학습 데이터 파일 경로
        target_column (str): 타겟 변수 컬럼 이름
        drop_columns (list): 제외할 컬럼 이름 리스트
        model_save_path (str): 최적 모델 저장 경로
        n_trials (int): Optuna 최적화 반복 횟수
        k_folds (int): K-fold 분할 수

    Returns:
        best_model: 최적 모델
        best_trial: Optuna의 최적 trial
    """
    # 데이터 로드 및 전처리
    raw_train = pd.read_csv(data_path)
    preprocessing_train = DataPreprocessor(raw_train).drop(columns=drop_columns)

    # 특성과 타겟 분리
    X = preprocessing_train.drop(columns=target_column).values
    y = preprocessing_train[target_column].values

    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optuna 최적화 실행
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X_scaled, y, k_folds=k_folds), n_trials=n_trials)

    # 최적 trial 및 모델
    best_trial = study.best_trial
    best_model_name = best_trial.params["model"]

    if best_model_name == "RandomForest":
        best_model = RandomForestRegressor(
            n_estimators=best_trial.params["n_estimators"],
            max_depth=best_trial.params["max_depth"],
            random_state=42
        )
    elif best_model_name == "ExtraTrees":
        best_model = ExtraTreesRegressor(
            n_estimators=best_trial.params["n_estimators"],
            max_depth=best_trial.params["max_depth"],
            random_state=42
        )
    elif best_model_name == "XGBoost":
        best_model = XGBRegressor(
            n_estimators=best_trial.params["n_estimators"],
            max_depth=best_trial.params["max_depth"],
            learning_rate=best_trial.params["learning_rate"],
            random_state=42
        )
    elif best_model_name == "LightGBM":
        best_model = LGBMRegressor(
            n_estimators=best_trial.params["n_estimators"],
            max_depth=best_trial.params["max_depth"],
            learning_rate=best_trial.params["learning_rate"],
            random_state=42,
            verbose=-1
        )
    elif best_model_name == "CatBoost":
        best_model = CatBoostRegressor(
            iterations=best_trial.params["iterations"],
            depth=best_trial.params["depth"],
            learning_rate=best_trial.params["learning_rate"],
            random_state=42,
            verbose=0
        )

    # 최적 모델 학습 및 저장
    best_model.fit(X_scaled, y)
    joblib.dump(best_model, model_save_path)
    print(f"Best model saved to {model_save_path}")

    # 스케일러 저장
    scaler_save_path = model_save_path.replace(".joblib", "_scaler.joblib")
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Trial Params: {best_trial.params}")
    print(f"Best RMSE: {best_trial.value}")

    return best_model, best_trial

# 파라미터 설정
data_path = 'EV_Cost_data/train.csv'
target_column = "가격(백만원)"
drop_columns = ['ID', '가격구간']
model_save_path = 'model_list/model/best_model_kfold_optuna.joblib'

# 실행
best_model, best_trial = train_with_kfold_and_save(data_path, target_column, drop_columns, model_save_path, n_trials=50, k_folds=5)
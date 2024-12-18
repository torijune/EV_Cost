import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, StackingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import joblib

from preprocessor import main_Mean as DataPreprocessor

def train_and_save_model(data_path, target_column, drop_columns, model_save_path):
    """
    학습 데이터를 사용해 여러 모델을 학습하고 최적의 모델을 저장합니다.

    Parameters:
        data_path (str): 학습 데이터 파일 경로
        target_column (str): 타겟 변수 컬럼 이름
        drop_columns (list): 제외할 컬럼 이름 리스트
        model_save_path (str): 최적 모델 저장 경로

    Returns:
        results_df (pd.DataFrame): 모델 성능 결과
        best_model_name (str): 최적 모델 이름
    """
    
    # 데이터 로드 및 전처리
    raw_train = pd.read_csv(data_path)
    preprocessing_train = DataPreprocessor(raw_train).drop(columns=drop_columns)

    # 특성과 타겟 분리
    X = preprocessing_train.drop(columns=target_column)
    y = preprocessing_train[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 데이터 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 베이스 모델 리스트
    base_models = {
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100),
        "ExtraTrees": ExtraTreesRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
        "CatBoost": CatBoostRegressor(verbose=0)
    }

    # Stacking 모델 정의
    stacking_estimators = [
        ('RandomForest', RandomForestRegressor(random_state=42, n_estimators=50)),
        ('GradientBoosting', GradientBoostingRegressor(random_state=42)),
        ('LightGBM', LGBMRegressor(random_state=42, verbose=-1)),
        ('XGBoost', XGBRegressor(random_state=42))
    ]
    stacking_model = StackingRegressor(
        estimators=stacking_estimators,
        final_estimator=RandomForestRegressor(random_state=42)
    )

    # Voting 모델 정의
    voting_model = VotingRegressor(estimators=[
        ('RandomForest', RandomForestRegressor(random_state=42, n_estimators=50)),
        ('GradientBoosting', GradientBoostingRegressor(random_state=42)),
        ('LightGBM', LGBMRegressor(random_state=42, verbose=-1)),
        ('XGBoost', XGBRegressor(random_state=42))
    ])

    # 모든 모델 추가
    models = base_models.copy()
    models["Stacking"] = stacking_model
    models["Voting"] = voting_model

    best_model = None
    best_rmse = float("inf")
    results = []

    # 각 모델 학습 및 평가
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # RMSE 계산

        results.append({"Model": name, "RMSE": rmse})

        # 최적 모델 선택
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = model

    # 결과 DataFrame 생성
    results_df = pd.DataFrame(results)

    # 최적 모델 저장
    joblib.dump(best_model, model_save_path)
    print(f"Best model saved to {model_save_path}")

    # 스케일러 저장
    scaler_save_path = model_save_path.replace(".joblib", "_scaler.joblib")
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

    print("\nTraining Results:")
    print(results_df)
    print(f"\nBest Model: {best_model.__class__.__name__} with RMSE: {best_rmse}")

    return results_df, best_model.__class__.__name__

# 파라미터 설정
data_path = 'EV_Cost_data/train.csv'
target_column = "가격(백만원)"
drop_columns = ['ID', '가격구간']
model_save_path = 'model_list/model/best_model_a.joblib'

# 실행
results, best_model_name = train_and_save_model(data_path, target_column, drop_columns, model_save_path)
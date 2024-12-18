import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import joblib

from preprocessor import main_Mean as DataPreprocessor

def train_models_per_manufacturer(data_path, target_column, drop_columns, model_save_dir):
    """
    제조사별로 개별 모델을 학습하고 저장합니다.

    Parameters:
        data_path (str): 학습 데이터 파일 경로
        target_column (str): 타겟 변수 컬럼 이름
        drop_columns (list): 제외할 컬럼 이름 리스트
        model_save_dir (str): 모델 저장 디렉토리 경로

    Returns:
        results_df (pd.DataFrame): 제조사별 모델 성능 결과
    """

    # 디렉토리 생성
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    # 데이터 로드 및 전처리
    raw_train = pd.read_csv(data_path)
    preprocessing_train = DataPreprocessor(raw_train).drop(columns=drop_columns)

    # 제조사 리스트
    manufacturers = preprocessing_train['제조사'].unique()

    results = []

    # 각 제조사별로 모델 학습
    for manufacturer in manufacturers:
        # 제조사별 데이터 필터링
        manufacturer_data = preprocessing_train[preprocessing_train['제조사'] == manufacturer]
        
        # 특성과 타겟 분리
        X = manufacturer_data.drop(columns=target_column)
        y = manufacturer_data[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 데이터 스케일링
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 모델 학습
        model = LGBMRegressor(random_state=42, verbose=-1)
        model.fit(X_train_scaled, y_train)

        # 예측 및 성능 평가
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # RMSE 계산

        # 모델 및 스케일러 저장
        model_path = os.path.join(model_save_dir, f"{manufacturer}_model.joblib")
        scaler_path = os.path.join(model_save_dir, f"{manufacturer}_scaler.joblib")

        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        print(f"Model for manufacturer '{manufacturer}' saved to {model_path}")
        print(f"Scaler for manufacturer '{manufacturer}' saved to {scaler_path}")

        # 결과 저장
        results.append({"Manufacturer": manufacturer, "RMSE": rmse})

    # 결과 DataFrame 생성
    results_df = pd.DataFrame(results)
    print("\nTraining Results:")
    print(results_df)

    return results_df

# 파라미터 설정
data_path = 'EV_Cost_data/train.csv'
target_column = "가격(백만원)"
drop_columns = ['ID', '가격구간']
model_save_dir = 'model_list/manufacturer_models'

# 실행
results_df = train_models_per_manufacturer(data_path, target_column, drop_columns, model_save_dir)
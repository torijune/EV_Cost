import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from pytorch_tabnet.tab_model import TabNetRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import numpy as np

from preprocessor import main as DataPreprocessor

raw_train = pd.read_csv('EV_Cost_data/train.csv')
drop_columns = 'ID'
preprocessing_train = DataPreprocessor(raw_train).drop(columns = drop_columns)

# 특성과 타겟 분리
X = preprocessing_train.drop(columns="가격(백만원)").to_numpy()
y = preprocessing_train["가격(백만원)"].to_numpy().reshape(-1, 1)  # 타겟을 2D로 변환

# 학습 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TabNet Regressor 추가
tabnet_model = TabNetRegressor(verbose=0, seed=42)

# TabNet 학습
tabnet_model.fit(
    X_train, y_train,  # 타겟이 2D 배열로 입력됨
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],
    max_epochs=100,
    patience=10,
    batch_size=32
)

# 예측
y_pred = tabnet_model.predict(X_test)

# 평가
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"TabNet Model - MSE: {mse}, R^2: {r2}")

# 성능 개 쓰레기
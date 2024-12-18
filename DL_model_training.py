import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
from preprocessor import main as DataPreprocessor


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class SimpleRegressionNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleRegressionNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

def train_neural_network(X_train, y_train, X_test, y_test, device, epochs=50, batch_size=32):
    input_dim = X_train.shape[1]
    model = SimpleRegressionNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 데이터셋 및 데이터로더 설정
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 학습
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()

    # 평가
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(test_tensor).squeeze().cpu().numpy()

    return model, predictions

def train_and_save_model(data_path, target_column, drop_columns, model_save_path):
    # 데이터 로드 및 전처리
    raw_train = pd.read_csv(data_path)
    preprocessing_train = DataPreprocessor(raw_train).drop(columns=drop_columns)

    # 특성과 타겟 분리
    X = preprocessing_train.drop(columns=target_column)
    y = preprocessing_train[target_column]

    # 학습 데이터와 테스트 데이터 분리
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 스케일링
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 모델 리스트
    models = {
        "XGBoost": XGBRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42)
    }

    # PyTorch에서 MPS 설정
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models["NeuralNetwork"] = "NeuralNetwork"

    best_model = None
    best_rmse = float("inf")
    results = []

    for name, model in models.items():
        if name == "NeuralNetwork":
            # PyTorch 학습
            nn_model, predictions = train_neural_network(
                X_train_scaled, y_train, X_test_scaled, y_test, device, epochs=50, batch_size=32
            )
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
        else:
            # 기존 머신러닝 모델 학습
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({"Model": name, "RMSE": rmse})

        # 최적 모델 선택
        if rmse < best_rmse:
            best_rmse = rmse
            best_model = nn_model if name == "NeuralNetwork" else model

    # 결과 DataFrame 생성
    results_df = pd.DataFrame(results)

    # 최적 모델 저장
    joblib.dump(best_model, model_save_path)
    print(f"Best model saved to {model_save_path}")

    # 스케일러 저장
    scaler_save_path = model_save_path.replace(".joblib", "_scaler.joblib")
    joblib.dump(scaler, scaler_save_path)
    print(f"Scaler saved to {scaler_save_path}")

    return results_df, best_model.__class__.__name__


data_path = 'EV_Cost_data/train.csv'  # 학습 데이터 경로
target_column = "가격(백만원)"          # 타겟 열 이름
drop_columns = ['ID', '가격구간']       # 제거할 열
model_save_path = 'best_model.joblib' # 모델 저장 경로

# 함수 호출
results, best_model_name = train_and_save_model(data_path, target_column, drop_columns, model_save_path)
print(results)
print(f"Best Model: {best_model_name}")
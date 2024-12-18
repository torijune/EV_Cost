import pandas as pd
import joblib
from test_preprocessing import main as test_preprocessor

def load_and_predict(model_path, test_data_path, target_column=None):
    # 모델 및 스케일러 로드
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # 테스트 데이터 로드
    test_data = pd.read_csv(test_data_path)
    test_data = test_preprocessor(test_data)
    if target_column:
        test_features = test_data.drop(columns=target_column)
    else:
        test_features = test_data

    # 스케일링
    test_features_scaled = scaler.transform(test_features)

    # 예측
    predictions = model.predict(test_features_scaled)
    test_data["Predicted"] = predictions

    return test_data

from model_prediction import load_and_predict

model_path = 'model_list/model/best_model_a.joblib'
scaler_path = 'model_list/scaler/best_model_a_scaler.joblib'
test_data_path = 'EV_Cost_data/test.csv'

predictions = load_and_predict(model_path, test_data_path)
submission = pd.read_csv('EV_Cost_data/sample_submission.csv')
submission["가격(백만원)"] = predictions["Predicted"].values[:len(submission)]

submission.to_csv('prediction_list/final_predictions_a.csv', index=False)
import pandas as pd
import joblib
from test_preprocessing import main as test_preprocessor
import os

def load_and_predict_per_manufacturer(model_dir, test_data_path, submission_path, output_path):
    """
    제조사별 모델을 사용하여 테스트 데이터를 예측하고, 결과를 저장합니다.

    Parameters:
        model_dir (str): 제조사별 모델과 스케일러가 저장된 디렉토리 경로
        test_data_path (str): 테스트 데이터 파일 경로
        submission_path (str): 기존 submission 파일 경로
        output_path (str): 결과를 저장할 파일 경로
    """
    # 모델 및 스케일러 로드 경로 확인
    model_files = [f for f in os.listdir(model_dir) if f.endswith("_model.joblib")]
    scaler_files = [f.replace("_model.joblib", "_scaler.joblib") for f in model_files]
    
    # 제조사별 모델 및 스케일러 로드
    models = {}
    scalers = {}
    for model_file, scaler_file in zip(model_files, scaler_files):
        manufacturer = model_file.split("_")[0]
        models[manufacturer] = joblib.load(os.path.join(model_dir, model_file))
        scalers[manufacturer] = joblib.load(os.path.join(model_dir, scaler_file))
    
    # 테스트 데이터 로드 및 전처리
    test_data = pd.read_csv(test_data_path)
    test_data = test_preprocessor(test_data)
    test_data["Predicted"] = 0  # 초기화

    # 각 제조사에 맞는 모델로 예측
    for manufacturer in models.keys():
        # 제조사별 데이터 필터링
        manufacturer_data = test_data[test_data['제조사'] == manufacturer]
        if manufacturer_data.empty:
            continue
        
        # 제조사별 특성 추출
        test_features = manufacturer_data.drop(columns=["ID", "Predicted"])
        test_features_scaled = scalers[manufacturer].transform(test_features)
        
        # 예측
        predictions = models[manufacturer].predict(test_features_scaled)
        test_data.loc[test_data['제조사'] == manufacturer, "Predicted"] = predictions

    # 기존 submission 파일 로드
    submission = pd.read_csv(submission_path)
    submission["가격(백만원)"] = test_data["Predicted"].values[:len(submission)]

    # 결과 저장
    submission.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

# 파라미터 설정
model_dir = 'model_list/manufacturer_models'
test_data_path = 'EV_Cost_data/test.csv'
submission_path = 'EV_Cost_data/sample_submission.csv'
output_path = 'prediction_list/final_predictions_per_manufacturer.csv'

# 실행
load_and_predict_per_manufacturer(model_dir, test_data_path, submission_path, output_path)
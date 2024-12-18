import pandas as pd
from target_preprocessing import categorize_by_mean_std
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 차량 상태에서 주행 거리가 비이상적으로 높거나 낮은 데이터들 전처리
def preprocessing_KM(df, mileage_column='주행거리(km)', year_column='연식(년)', new_column='연간_주행거리'):
    df.loc[(df['차량상태'] == 'Nearly New') & (df[mileage_column] > 50000), '차량상태'] = 'Pre-Owned'
    df.loc[(df['차량상태'] == 'Pre-Owned') & (df[mileage_column] <= 50000), '차량상태'] = 'Nearly New'
    df[new_column] = df[mileage_column] / (df[year_column] + 1)
    return df

# train data에서 가격을 활용하여 제조사들을 클러스터링 하여 사용
def preprocessing_price(df):
    def categorize_manufacturer(row):
        if (row['고가'] > 0.2) & (row['고가'] < 0.7):
            return '고가 제조사'
        elif (row['저가'] < 0.35) & (row['중가'] > 0.5):
            return '중가 제조사'
        elif row['고가'] >= 0.7:
            return '초고가 제조사'
        else:
            return '저가 제조사'

    def make_maun_cate(df):
        price_distribution = df.groupby('제조사')['가격구간'].value_counts(normalize=True).unstack(fill_value=0)
        price_distribution['제조사_카테고리'] = price_distribution.apply(categorize_manufacturer, axis=1)
        df = df.merge(price_distribution[['제조사_카테고리']], on='제조사', how='left')
        return df

    df['가격구간'] = categorize_by_mean_std(df, column="가격(백만원)")
    df = make_maun_cate(df)
    return df

# 범주형 변수 및 인코딩이 필요한 변수들을 라벨 인코딩
def encoding(df):
    encoding_col = df.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in encoding_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    return df

import pandas as pd

def fill_with_mean(df):
    # 모델별 평균 계산
    model_means = df.groupby('모델')['배터리용량'].mean()

    # 널값을 모델별 평균으로 채우기
    for index, row in df.iterrows():
        if pd.isna(row['배터리용량']):
            model = row['모델']
            if model in model_means.index:
                df.loc[index, '배터리용량'] = model_means[model]
            else:
                df.loc[index, '배터리용량'] = df['배터리용량'].mean()  # 전체 평균 사용

    return df

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

def fill_with_ml(df):
    # 배터리 관련 컬럼 추출
    battery_col = ['연간_주행거리', '보증기간(년)', '차량상태', '구동방식', '배터리용량', '가격(백만원)']
    battery_pred_df = df[battery_col]

    # 학습 데이터와 테스트 데이터 분리
    train_bat = battery_pred_df[battery_pred_df['배터리용량'].notnull()]
    test_bat = battery_pred_df[battery_pred_df['배터리용량'].isnull()]

    # 머신러닝 모델 학습
    X_train = train_bat.drop(columns=['배터리용량'])
    y_train = train_bat['배터리용량']
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 예측 수행
    X_test = test_bat.drop(columns=['배터리용량'])
    predicted_values = model.predict(X_test)

    # 예측값으로 널값 채우기
    df.loc[df['배터리용량'].isnull(), '배터리용량'] = predicted_values

    return df

def main_ML(df):
    df = preprocessing_KM(df)
    df = preprocessing_price(df)
    df = encoding(df)
    df = fill_with_ml(df)
    return df

def main_Mean(df):
    df = preprocessing_KM(df)
    df = preprocessing_price(df)
    # df = encoding(df)
    df = fill_with_mean(df)
    return df

df = pd.read_csv('EV_Cost_data/train.csv')
print(main_Mean(df).isnull().sum().sum())
print(main_Mean(df).head())
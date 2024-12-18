import pandas as pd
import numpy as np
from preprocessor import fill_with_mean, preprocessing_KM, encoding

def manu_cate(df):
    # 모든 제조사 값을 포함한 매핑 딕셔너리
    manufacturer_category = {
        0: 0,
        1: 2,
        2: 2,
        3: 1,
        4: 3,
        5: 2,
        6: 2
    }
    
    # 제조사 카테고리 매핑
    df["제조사_카테고리"] = df["제조사"].map(manufacturer_category)
    return df

def main(df):
    df = preprocessing_KM(df)
    df = encoding(df)
    df = fill_with_mean(df)
    df = df.drop(columns = 'ID')
    df = manu_cate(df)
    # print(df.head())
    return df

print(main(pd.read_csv('EV_Cost_data/test.csv')).head(2))
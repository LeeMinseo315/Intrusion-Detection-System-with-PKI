
import pandas as pd

import numpy as np

from sklearn.ensemble import IsolationForest

import joblib

print("데이터 로딩 중...")

df = pd.read_csv('/mnt/c/Users/은서/Desktop/캡스톤디자인2/features_smote_tls.csv')

features = ['Flow Duration', 'Total Length of Fwd Packets', 

            'Total Length of Bwd Packets',

            'tls_version', 'cipher_strength', 'tls_established']

# 정상 데이터로 IF 학습

X_normal = df[df['Label'] == 0][features].fillna(0)

iso = IsolationForest(contamination=0.03, random_state=42)

iso.fit(X_normal)

# 규칙 기반 필터 + IF 결합 탐지

def detect(row):

    # 1차: 규칙 기반 (tls_established=0 이면 무조건 이상치)

    if row['tls_established'] == 0:

        return -1

    # 2차: IF 모델로 판단

    x = [[row[f] for f in features]]

    return iso.predict(x)[0]

print("탐지 중...")

X_all = df[features].fillna(0)

preds = X_all.apply(detect, axis=1)

# 시나리오 A 탐지율

y_true = df.apply(lambda x: -1 if (x['Label'] != 0 and x['tls_established'] == 0) else 1, axis=1)

detected = ((preds == -1) & (y_true == -1)).sum()

total_a = (y_true == -1).sum()

print(f"정상 판단 : {(preds == 1).sum()}")

print(f"이상 판단 : {(preds == -1).sum()}")

print(f"시나리오A 탐지율 : {detected/total_a:.4f} ({detected}/{total_a})")

joblib.dump(iso, 'ml/isolation_forest_tls.pkl')

import pickle

with open('ml/iso_rule_filter.pkl', 'wb') as f:

    pickle.dump({'contamination': 0.5, 'rule': 'tls_established==0'}, f)

print("저장 완료")


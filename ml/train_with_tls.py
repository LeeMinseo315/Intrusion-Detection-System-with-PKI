
import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score

import joblib

import json

print("데이터 로딩 중...")

df = pd.read_csv('/mnt/c/Users/은서/Desktop/캡스톤디자인2/features_smote_tls.csv')

with open('schema.json', 'r') as f:

    schema = json.load(f)

features_old = schema['features']

features_new = schema['features'] + ['tls_version', 'cipher_strength', 'tls_established']

X = df[features_new]

y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 기존 모델 성능

print("\n기존 RF 모델 (TLS 피처 없음)")

rf_old = joblib.load('ml/model_rf.pkl')

old_pred = rf_old.predict(X_test[features_old])

print(f"Accuracy : {accuracy_score(y_test, old_pred):.4f}")

print(f"F1-score : {f1_score(y_test, old_pred, average='weighted'):.4f}")

# TLS 피처 추가 모델

print("\nTLS 피처 추가 RF 모델 학습 중...")

rf_new = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

rf_new.fit(X_train, y_train)

new_pred = rf_new.predict(X_test)

print(f"Accuracy : {accuracy_score(y_test, new_pred):.4f}")

print(f"F1-score : {f1_score(y_test, new_pred, average='weighted'):.4f}")

# 피처 중요도

print("\nTLS 피처 중요도:")

importances = pd.Series(rf_new.feature_importances_, index=features_new)

print(importances[['tls_version', 'cipher_strength', 'tls_established']])

joblib.dump(rf_new, 'ml/model_rf_tls.pkl')

print("\n저장 완료: model_rf_tls.pkl")


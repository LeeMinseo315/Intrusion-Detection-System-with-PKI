
import pandas as pd

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

from sklearn.metrics import f1_score, accuracy_score

import joblib

import json

print("데이터 로딩 중...")

df = pd.read_csv('/mnt/c/Users/은서/Desktop/캡스톤디자인2/features_smote_tls.csv')

with open('schema.json', 'r') as f:

    schema = json.load(f)

features_new = schema['features'] + ['tls_version', 'cipher_strength', 'tls_established']

X = df[features_new]

y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM은 데이터 많으면 느려서 샘플링

print("\nTLS 피처 추가 SVM 학습 중...")

svm = SVC(kernel='rbf', random_state=42)

svm.fit(X_train[:10000], y_train[:10000])

svm_pred = svm.predict(X_test[:10000])

print(f"Accuracy : {accuracy_score(y_test[:10000], svm_pred):.4f}")

print(f"F1-score : {f1_score(y_test[:10000], svm_pred, average='weighted'):.4f}")

joblib.dump(svm, 'ml/model_svm_tls.pkl')

print("\n저장 완료: model_svm_tls.pkl")


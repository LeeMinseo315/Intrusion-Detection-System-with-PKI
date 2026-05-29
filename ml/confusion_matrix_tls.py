
import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

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

label_names = ['Benign', 'DDoS', 'PortScan', 'Web Attack', 'Bot', 'Infiltration']

# RF 혼동 행렬

print("\nRandom Forest 혼동 행렬...")

rf = joblib.load('ml/model_rf_tls.pkl')

rf_pred = rf.predict(X_test)

rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(10, 8))

disp = ConfusionMatrixDisplay(confusion_matrix=rf_cm, display_labels=label_names[:len(rf_cm)])

disp.plot(cmap='Blues', values_format='d')

plt.title('Random Forest (TLS 피처 추가) - 혼동 행렬')

plt.tight_layout()

plt.savefig('ml/confusion_matrix_rf_tls.png', dpi=150)

print("저장 완료: confusion_matrix_rf_tls.png")

# SVM 혼동 행렬

print("\nSVM 혼동 행렬...")

svm = joblib.load('ml/model_svm_tls.pkl')

svm_pred = svm.predict(X_test[:10000])

svm_cm = confusion_matrix(y_test[:10000], svm_pred)

plt.figure(figsize=(10, 8))

disp = ConfusionMatrixDisplay(confusion_matrix=svm_cm, display_labels=label_names[:len(svm_cm)])

disp.plot(cmap='Oranges', values_format='d')

plt.title('SVM (TLS 피처 추가) - 혼동 행렬')

plt.tight_layout()

plt.savefig('ml/confusion_matrix_svm_tls.png', dpi=150)

print("저장 완료: confusion_matrix_svm_tls.png")

print("\n완료!")


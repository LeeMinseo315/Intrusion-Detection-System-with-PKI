
import pandas as pd

import numpy as np

import time

import psutil

import os

import joblib

import json

# psutil 설치 확인

try:

    import psutil

except:

    os.system('pip install psutil')

    import psutil

print("데이터 로딩 중...")

df = pd.read_csv('/mnt/c/Users/은서/Desktop/캡스톤디자인2/features_smote_tls.csv')

with open('schema.json', 'r') as f:

    schema = json.load(f)

features_old = schema['features']

features_new = schema['features'] + ['tls_version', 'cipher_strength', 'tls_established']

# 10,000개 샘플 추출

X_test = df[features_new].sample(10000, random_state=42)

X_test_old = df[features_old].sample(10000, random_state=42)

def benchmark(model_name, model, X):

    process = psutil.Process(os.getpid())

    

    # 메모리 측정 시작

    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    

    # 추론 시간 측정

    start = time.perf_counter()

    model.predict(X)

    end = time.perf_counter()

    

    # 메모리 측정 종료

    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    

    elapsed = end - start

    mem_used = mem_after - mem_before

    throughput = 10000 / elapsed

    

    print(f"\n{'='*50}")

    print(f"모델명       : {model_name}")

    print(f"추론 시간    : {elapsed:.4f}초")

    print(f"처리량       : {throughput:.0f} 건/초")

    print(f"메모리 변화  : {mem_used:.2f} MB")

    print(f"{'='*50}")

    

    return {

        'model': model_name,

        'time': round(elapsed, 4),

        'throughput': round(throughput, 0),

        'memory_mb': round(mem_used, 2)

    }

results = []

# RF (기존)

print("\nRandom Forest (TLS 피처 없음) 벤치마크...")

rf_old = joblib.load('ml/model_rf.pkl')

results.append(benchmark('RF (기존)', rf_old, X_test_old))

# RF (TLS 피처 추가)

print("\nRandom Forest (TLS 피처 추가) 벤치마크...")

rf_new = joblib.load('ml/model_rf_tls.pkl')

results.append(benchmark('RF (TLS 추가)', rf_new, X_test))

# SVM (TLS 피처 추가)

print("\nSVM (TLS 피처 추가) 벤치마크...")

svm = joblib.load('ml/model_svm_tls.pkl')

results.append(benchmark('SVM (TLS 추가)', svm, X_test[:10000]))

# 결과 비교표

print("\n\n최종 벤치마크 비교표")

print(f"{'모델':<25} {'추론시간(초)':<15} {'처리량(건/초)':<15} {'메모리(MB)':<10}")

print("-" * 65)

for r in results:

    print(f"{r['model']:<25} {r['time']:<15} {r['throughput']:<15} {r['memory_mb']:<10}")

# CSV 저장

result_df = pd.DataFrame(results)

result_df.to_csv('ml/benchmark_results.csv', index=False)

print("\n저장 완료: benchmark_results.csv")


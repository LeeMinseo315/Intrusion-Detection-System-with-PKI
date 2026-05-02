import sys
import joblib
import pandas as pd
import json
import os

def load_artifacts():
    # 학습된 아티팩트들 로드
    model = joblib.load("model_rf.pkl")
    scaler = joblib.load("scaler.pkl")
    le = joblib.load("label_encoder.pkl")
    
    # [추가] 스마 로드 (피처 순서 보장용)
    with open("schema.json", "r", encoding="utf-8") as f:
        schema = json.load(f)
    features = schema["features"]
    
    return model, scaler, le, features

def predict(csv_path):
    if not os.path.exists(csv_path):
        print(f"❌ 에러: 파일을 찾을 수 없습니다. ({csv_path})")
        return

    print("📦 모델, 스케일러 및 스키마 로딩 중...")
    model, scaler, le, features = load_artifacts()

    # ------------------- 데이터 로드 -------------------
    X_raw = pd.read_csv(csv_path)
    
    # ------------------- 스키마 기반 정렬 및 필터링 -------------------
    # [수정] schema.json에 정의된 15개 피처만, 정의된 순서대로 정확히 추출합니다.
    try:
        X_new = X_raw[features].copy()
    except KeyError as e:
        missing_cols = set(features) - set(X_raw.columns)
        print(f"❌ 에러: CSV 파일에 필요한 피처가 부족합니다.")
        print(f"누락된 피처: {missing_cols}")
        sys.exit(1)

    # ------------------- 전처리 (결측치 처리 및 변환) -------------------
    # Nan 혹은 Infinity 값이 있으면 모델이 터지므로 0으로 채워줍니다.
    X_new = X_new.fillna(0).replace([float('inf'), float('-inf')], 0)

    # ------------------- 스케일링 -------------------
    # 이제 순서와 개수가 보장된 상태에서 스케일링을 진행합니다.
    X_scaled_arr = scaler.transform(X_new)

    # ------------------- 예측 -------------------
    preds = model.predict(X_scaled_arr)
    pred_probs = model.predict_proba(X_scaled_arr)

    # ------------------- 결과 정리 -------------------
    pred_labels = le.inverse_transform(preds)
    confidences = pred_probs.max(axis=1) * 100

    report = pd.DataFrame({
        "Row_Index": X_raw.index,
        "Label_Name": pred_labels,
        "Confidence_Pct": confidences.round(2)
    })

    # ------------------- 출력 -------------------
    print(f"\n🔍 [예측 결과 보고서: {csv_path}]")
    print("=" * 55)
    print(f"총 데이터 개수: {len(report)} 행")
    print("-" * 55)
    print("탐지된 라벨별 분포 (개수):")
    label_counts = report["Label_Name"].value_counts()
    for label, count in label_counts.items():
        print(f" - {label:<15}: {count} 건")
    
    print("-" * 55)
    print("상위 10개 행 상세 결과:")
    print(report.head(10).to_string(index=False))
    print("=" * 55)

    # (옵션) 결과 저장
    # output_path = "prediction_result.csv"
    # report.to_csv(output_path, index=False)
    # print(f"✅ 결과가 {output_path}에 저장되었습니다.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("사용법: python predict.py <csv_path>")
        sys.exit(1)
    predict(sys.argv[1])

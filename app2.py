from flask import Flask, jsonify, render_template
import ssl
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime

import requests

WEBHOOK_URL = "https://discord.com/api/webhooks/1512423070957047811/t86RDfs0_XjFaE5W9BYJaNTjjwyDA1xnNqBVU0btoW01ng--v6PqEHlRpbZN2EAL4xM3"

app = Flask(__name__)

def send_alert(prediction, anomaly_score, timestamp):

    message = {

        "content": f"**[보안 경고]** 공격 탐지!\n"

                   f"• 공격 유형: {prediction}\n"

                   f"• 이상 점수: {anomaly_score}\n"

                   f"• 탐지 시간: {timestamp}"

    }

    requests.post(WEBHOOK_URL, json=message)

# ==========================================
# 모델 및 데이터 로드
# ==========================================
rf_model = joblib.load('model_rf_tls.pkl')
iso_model = joblib.load('isolation_forest_tls.pkl')

features_15 = [
    'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
    'Fwd Packet Length Max', 'Fwd Packet Length Mean',
    'Bwd Packet Length Max', 'Bwd Packet Length Mean',
    'Flow IAT Mean', 'Flow IAT Max', 'Flow Bytes/s',
    'Flow Packets/s', 'SYN Flag Count', 'ACK Flag Count'
]
features_18 = features_15 + ['tls_version', 'cipher_strength', 'tls_established']
features_iso = ['Flow Duration', 'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
                'tls_version', 'cipher_strength', 'tls_established']

label_map = {0: 'BENIGN', 1: 'DDoS', 2: 'PortScan', 3: 'Web Attack'}

# 데이터 로드
test_df = pd.read_csv('test_data.csv')
X_test = test_df[features_18].reset_index(drop=True)
y_test = test_df['Label'].reset_index(drop=True)

# ==========================================
# predictions.json 생성 함수
# ==========================================
def generate_predictions(n_samples=100):
    # 수정 1: 층화 샘플링
    tls_0 = X_test[X_test['tls_established'] == 0]
    tls_1 = X_test[X_test['tls_established'] == 1]
    n_tls_0 = int(n_samples * len(tls_0) / len(X_test))
    n_tls_1 = n_samples - n_tls_0
    idx_0 = tls_0.sample(n=n_tls_0, random_state=42).index
    idx_1 = tls_1.sample(n=n_tls_1, random_state=42).index
    idx = idx_0.append(idx_1)
    samples = X_test.loc[idx].reset_index(drop=True)
    y_samples = y_test.loc[idx].reset_index(drop=True)

    # RF 예측
    X_rf = samples[features_18]
    pred_classes = rf_model.predict(X_rf)
    pred_probas = rf_model.predict_proba(X_rf)

    # ISO 예측
    X_iso = samples[features_iso]
    rule_anomaly = (X_iso['tls_established'] == 0).astype(int)
    iso_raw = iso_model.predict(X_iso)
    if_anomaly = (pd.Series(iso_raw) == -1).astype(int)
    iso_preds_combined = ((rule_anomaly.values == 1) | (if_anomaly.values == 1))
    iso_scores = iso_model.decision_function(X_iso)

    predictions = []
    for i in range(n_samples):
        predictions.append({
            "id": i + 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prediction": label_map[int(pred_classes[i])],
            "confidence": round(float(max(pred_probas[i])), 4),
            "is_anomaly": bool(iso_preds_combined[i]),
            "anomaly_score": round(float(iso_scores[i]), 4),
            # 수정 2: tls_established 추가
            "tls_established": int(samples.iloc[i]['tls_established']),
            "class_probabilities": {
                "BENIGN": round(float(pred_probas[i][0]), 4),
                "DDoS": round(float(pred_probas[i][1]), 4),
                "PortScan": round(float(pred_probas[i][2]), 4),
                "Web Attack": round(float(pred_probas[i][3]), 4)
            },
            "actual": label_map[int(y_samples.iloc[i])]
        })
        # 수정 3: send_alert for 루프 안으로 이동
        if pred_classes[i] != 0 or iso_preds_combined[i]:
            send_alert(
                label_map[int(pred_classes[i])],
                round(float(iso_scores[i]), 4),
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )

    with open('predictions.json', 'w') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"predictions.json 생성 완료! ({n_samples}개 샘플)")
    return predictions

# ==========================================
# Flask 엔드포인트
# ==========================================
@app.route("/")
def home():
    return "Hello mTLS NIDS"

@app.route('/predict', methods=['GET'])
def predict():
    predictions = generate_predictions(n_samples=100)
    return jsonify({
        "status": "success",
        "count": len(predictions),
        "predictions": predictions
    })

@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)
        return jsonify(predictions)
    except FileNotFoundError:
        return jsonify({"error": "predictions.json이 없어요. /predict 먼저 실행하세요!"}), 404

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/stats', methods=['GET'])
def stats():
    try:
        with open('predictions.json', 'r') as f:
            predictions = json.load(f)

        total = len(predictions)
        attack_count = sum(1 for p in predictions if p['prediction'] != 'BENIGN')
        anomaly_count = sum(1 for p in predictions if p['is_anomaly'])

        class_counts = {'BENIGN': 0, 'DDoS': 0, 'PortScan': 0, 'Web Attack': 0}
        for p in predictions:
            class_counts[p['prediction']] += 1

        return jsonify({
            "total": total,
            "attack_count": attack_count,
            "anomaly_count": anomaly_count,
            "class_distribution": class_counts,
            "attack_rate": round(attack_count / total * 100, 2)
        })
    except FileNotFoundError:
        return jsonify({"error": "predictions.json이 없어요!"}), 404
    
@app.route('/inject_spy', methods=['GET'])
def inject_spy():
    spy_df = pd.read_csv('spy_data.csv')
    X_rf = spy_df[features_18]
    X_iso = spy_df[features_iso]

    pred = rf_model.predict(X_rf)
    proba = rf_model.predict_proba(X_rf)
    iso_pred = iso_model.predict(X_iso)
    iso_score = iso_model.decision_function(X_iso)

    is_anomaly = bool(iso_pred[0] == -1)

    if is_anomaly:
        send_alert(
            "스파이 트래픽",
            round(float(iso_score[0]), 4),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    return jsonify({
        "rf_prediction": label_map[int(pred[0])],
        "rf_confidence": round(float(max(proba[0])), 4),
        "is_anomaly": is_anomaly,
        "anomaly_score": round(float(iso_score[0]), 4),
        "message": "내부 이상 행위 감지!" if is_anomaly else "정상"
    })

# ==========================================
# mTLS 설정
# ==========================================
if __name__ == "__main__":
    print("Flask 서버 시작 중...")
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(
        certfile="server_chain.crt",
        keyfile="server.key"
    )
    context.load_verify_locations(cafile="ca.crt")
    context.verify_mode = ssl.CERT_REQUIRED
    app.run(host="0.0.0.0", port=5000, ssl_context=context)
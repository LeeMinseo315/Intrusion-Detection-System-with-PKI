
import pandas as pd

import numpy as np

from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import LabelEncoder

import joblib

# conn.log 로드

conn = pd.read_csv('conn.log', sep='\t', comment='#', header=None,

    names=['ts','uid','orig_h','orig_p','resp_h','resp_p','proto','service',

           'duration','orig_bytes','resp_bytes','conn_state','local_orig',

           'local_resp','missed_bytes','history','orig_pkts','orig_ip_bytes',

           'resp_pkts','resp_ip_bytes','tunnel_parents','ip_proto'])

# ssl.log 로드

ssl = pd.read_csv('ssl.log', sep='\t', comment='#', header=None,

    names=['ts','uid','orig_h','orig_p','resp_h','resp_p','version','cipher',

           'curve','server_name','resumed','last_alert','next_protocol',

           'established','ssl_history','cert_chain_fps','client_cert_chain_fps',

           'sni_matches_cert'])

# 병합

merged = pd.merge(conn, ssl[['uid','version','cipher','established']], on='uid', how='inner')

# 피처 변환

merged['tls_version'] = merged['version'].map({'TLSv13': 1, 'TLSv12': 0}).fillna(-1)

merged['cipher_strength'] = merged['cipher'].map({

    'TLS_AES_256_GCM_SHA384': 3,

    'TLS_AES_128_GCM_SHA256': 2,

    'TLS_CHACHA20_POLY1305_SHA256': 2

}).fillna(0)

merged['tls_established'] = merged['established'].map({'T': 1, 'F': 0}).fillna(0)

merged['duration'] = pd.to_numeric(merged['duration'], errors='coerce').fillna(0)

merged['orig_bytes'] = pd.to_numeric(merged['orig_bytes'], errors='coerce').fillna(0)

merged['resp_bytes'] = pd.to_numeric(merged['resp_bytes'], errors='coerce').fillna(0)

features = ['duration', 'orig_bytes', 'resp_bytes', 'tls_version', 'cipher_strength', 'tls_established']

X = merged[features].fillna(0)

print(f"학습 데이터: {X.shape}")

# Isolation Forest 학습

iso = IsolationForest(contamination=0.05, random_state=42)

iso.fit(X)

scores = iso.decision_function(X)

predictions = iso.predict(X)

print(f"정상 샘플: {(predictions == 1).sum()}")

print(f"이상 샘플: {(predictions == -1).sum()}")

print(f"평균 이상치 점수: {scores.mean():.4f}")

joblib.dump(iso, 'isolation_forest_tls.pkl')

print("저장 완료: isolation_forest_tls.pkl")


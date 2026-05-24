import pandas as pd

# 병합된 데이터 로드
df = pd.read_csv('merged_tls.csv')

# TLS 피처 추출
df['tls_version'] = df['version'].map({'TLSv13': 1, 'TLSv12': 0}).fillna(-1)
df['cipher_strength'] = df['cipher'].map({
    'TLS_AES_256_GCM_SHA384': 3,
    'TLS_AES_128_GCM_SHA256': 2,
    'TLS_CHACHA20_POLY1305_SHA256': 2
}).fillna(0)
df['tls_established'] = df['established'].map({'T': 1, 'F': 0}).fillna(0)

# 최종 피처 선택
tls_features = df[['uid', 'orig_p', 'resp_p', 'duration', 
                    'orig_bytes', 'resp_bytes',
                    'tls_version', 'cipher_strength', 'tls_established']]

print("TLS 피처 추출 완료:")
print(tls_features.head())
print(f"\n피처 수: {len(tls_features.columns)}")

tls_features.to_csv('tls_features.csv', index=False)
print("저장 완료: tls_features.csv")

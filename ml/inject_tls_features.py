
import pandas as pd

import numpy as np

print("데이터 로딩 중...")

df = pd.read_csv('/mnt/c/Users/은서/Desktop/캡스톤디자인2//mnt/c/Users/은서/Desktop/캡스톤디자인2/features_smote.csv')

print(f"원본 데이터 shape: {df.shape}")

print(f"Label 분포:\n{df['Label'].value_counts()}")

# TLS 피처 초기화

df['tls_version'] = 0

df['cipher_strength'] = 0

df['tls_established'] = 0

# 정상 트래픽 (Label 0) → mTLS 정상 통신

df.loc[df['Label'] == 0, 'tls_version'] = 1

df.loc[df['Label'] == 0, 'cipher_strength'] = 3

df.loc[df['Label'] == 0, 'tls_established'] = 1

# 공격 트래픽 → 시나리오 A/B 50:50 분할

attack_idx = df[df['Label'] != 0].index

scenario_a = attack_idx[:len(attack_idx)//2]

scenario_b = attack_idx[len(attack_idx)//2:]

# 시나리오 A: 인증서 없는 외부 공격 (mTLS 차단)

df.loc[scenario_a, 'tls_version'] = 0

df.loc[scenario_a, 'cipher_strength'] = 0

df.loc[scenario_a, 'tls_established'] = 0

# 시나리오 B: 인증서 탈취한 내부 공격 (mTLS 통과)

df.loc[scenario_b, 'tls_version'] = 1

df.loc[scenario_b, 'cipher_strength'] = 3

df.loc[scenario_b, 'tls_established'] = 1

print(f"\nTLS 피처 추가 완료!")

print(f"최종 데이터 shape: {df.shape}")

print(f"컬럼: {list(df.columns)}")

df.to_csv('/mnt/c/Users/은서/Desktop/캡스톤디자인2//mnt/c/Users/은서/Desktop/캡스톤디자인2/features_smote_tls.csv', index=False)

print("저장 완료: /mnt/c/Users/은서/Desktop/캡스톤디자인2/features_smote_tls.csv")


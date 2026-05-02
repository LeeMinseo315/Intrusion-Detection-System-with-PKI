#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
import numpy as np
import pandas as pd
from pathlib import Path

# ---------- 1️⃣ 스키마 로드 ----------
SCHEMA_PATH = Path("schema.json")
if not SCHEMA_PATH.is_file():
    raise FileNotFoundError(f"⚠️ {SCHEMA_PATH} 를 찾을 수 없습니다.")

with SCHEMA_PATH.open(encoding="utf-8") as f:
    schema = json.load(f)

features = schema["features"]
print(f"🔎 로드된 피처({len(features)}개): {features}")

# ---------- 2️⃣ 각 피처별 값 범위 정의 (에러 해결 핵심!) ----------
# 민서님의 schema.json에 있는 15개 피처를 모두 포함했습니다.
RANGES = {
    "Flow Duration":            (0, 5000000),
    "Total Fwd Packets":        (0, 10000),
    "Total Backward Packets":   (0, 10000),
    "Total Length of Fwd Packets": (0, 50000000),
    "Total Length of Bwd Packets": (0, 50000000),
    "Fwd Packet Length Max":    (0, 1500),
    "Fwd Packet Length Mean":   (0, 1500),
    "Bwd Packet Length Max":    (0, 1500),
    "Bwd Packet Length Mean":   (0, 1500),
    "Flow IAT Mean":            (0.0, 5000.0),
    "Flow IAT Max":             (0.0, 10000.0),
    "Flow Bytes/s":             (0.0, 10000000.0),  # 새로 추가된 부분
    "Flow Packets/s":            (0.0, 5000.0),      # 새로 추가된 부분
    "SYN Flag Count":           (0, 1000),
    "ACK Flag Count":           (0, 1000),
}

# ---------- 3️⃣ 임의 데이터 생성 ----------
NUM_ROWS = 10
rows = []
for _ in range(NUM_ROWS):
    row = {}
    for ft in features:
        # RANGES에서 값을 가져오지 못할 경우를 대비한 안전망 추가
        if ft in RANGES:
            low, high = RANGES[ft]
        else:
            # 예상치 못한 피처가 있을 경우 기본값 세팅
            print(f"⚠️ 경고: {ft}의 범위를 알 수 없어 기본값(0-100)을 사용합니다.")
            low, high = (0, 100)
            
        if isinstance(low, int) and isinstance(high, int):
            value = random.randint(low, high)
        else:
            value = random.uniform(low, high)
        row[ft] = value
    rows.append(row)

df = pd.DataFrame(rows, columns=features)
df = df.replace([np.inf, -np.inf], 0).fillna(0)

# ---------- 4️⃣ CSV 저장 ----------
OUTPUT_PATH = Path("test_data.csv")
df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ {OUTPUT_PATH} 가 성공적으로 생성되었습니다! (행 수: {NUM_ROWS})")

import pandas as pd
import numpy as np
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier, IsolationForest, VotingClassifier
from sklearn.svm import SVC

# ==========================================
# 1. 데이터 로드
# ==========================================
print("데이터 로딩 중...")

files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
]

dfs = []
for f in files:
    df = pd.read_csv(f)
    df.columns = df.columns.str.strip()
    dfs.append(df)

df = pd.concat(dfs, ignore_index=True)
df['Label'] = df['Label'].str.strip()

# 라벨명 정리 (Web Attack 통합)
df['Label'] = df['Label'].str.replace(r'Web Attack .+ Brute Force', 'Web Attack', regex=True)
df['Label'] = df['Label'].str.replace(r'Web Attack .+ XSS', 'Web Attack', regex=True)
df['Label'] = df['Label'].str.replace(r'Web Attack .+ Sql Injection', 'Web Attack', regex=True)

# 필요한 컬럼만 사용
features = ['Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Total Length of Fwd Packets', 'Total Length of Bwd Packets',
            'Fwd Packet Length Max', 'Fwd Packet Length Mean',
            'Bwd Packet Length Max', 'Bwd Packet Length Mean',
            'Flow IAT Mean', 'Flow IAT Max', 'Flow Bytes/s',
            'Flow Packets/s', 'SYN Flag Count', 'ACK Flag Count']

df = df[features + ['Label']].dropna()
df = df[~df.isin([np.inf, -np.inf]).any(axis=1)]

print("원본 데이터 분포:")
print(df['Label'].value_counts())

# ==========================================
# 2. 샘플링 전략 적용
# ==========================================
print("\n샘플링 전략 적용 중...")

def undersample(df, label, n):
    cls = df[df['Label'] == label]
    return resample(cls, n_samples=n, random_state=42)

benign = undersample(df, 'BENIGN', 10000)
ddos = undersample(df, 'DDoS', 10000)
portscan = undersample(df, 'PortScan', 10000)
webattack = df[df['Label'] == 'Web Attack']  # 원본 그대로 (2180개)

df_balanced = pd.concat([benign, ddos, portscan, webattack], ignore_index=True)

print("\n샘플링 후 분포:")
print(df_balanced['Label'].value_counts())

# ==========================================
# 3. 라벨 인코딩
# ==========================================
label_map = {
    'BENIGN': 0,
    'DDoS': 1,
    'PortScan': 2,
    'Web Attack': 3,
}

df_balanced['Label'] = df_balanced['Label'].map(label_map)

X = df_balanced[features]
y = df_balanced['Label']

# ==========================================
# 4. Train/Test Split & 스케일링
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 5. SMOTE 적용
# ==========================================
print("\nSMOTE 적용 중...")

sampling_strategy = {
    0: 10000,  # BENIGN
    1: 10000,  # DDoS
    2: 10000,  # PortScan
    3: 8000,   # Web Attack (2180 → 8000)
}

smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=5, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

print("\nSMOTE 후 분포:")
print(pd.Series(y_train_res).value_counts())

# ==========================================
# 6. 모델 학습 및 평가
# ==========================================
target_names = ["BENIGN", "DDoS", "PortScan", "Web Attack"]

def evaluate_model(model, X_test, y_test, name):
    if name == 'Isolation Forest':
        raw_pred = model.predict(X_test)
        y_pred = np.where(raw_pred == 1, 0, 1)
        y_test_binary = np.where(y_test == 0, 0, 1)
        
        print(f"\n--- {name} 성능 리포트 (정상 vs 이상) ---")
        print(classification_report(y_test_binary, y_pred,
              target_names=["Normal", "Attack"]))
        
        # 혼동행렬 (이진)
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_test_binary, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=["Normal", "Attack"],
                    yticklabels=["Normal", "Attack"])
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'conf_matrix_{name}.png')
        plt.close()

    else:
        y_pred = model.predict(X_test)
        
        print(f"\n--- {name} 성능 리포트 (다중 분류) ---")
        print(classification_report(y_test, y_pred, target_names=target_names))

        # 다중 분류 혼동행렬
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'conf_matrix_{name}.png')
        plt.close()

        # 이진 변환 후 추가 비교용 리포트
        y_pred_binary = np.where(y_pred == 0, 0, 1)
        y_test_binary = np.where(y_test == 0, 0, 1)
        print(f"\n--- {name} 성능 리포트 (정상 vs 이상 변환) ---")
        print(classification_report(y_test_binary, y_pred_binary,
              target_names=["Normal", "Attack"]))

    print(f">>> {name} 혼동행렬 저장 완료")

# 모델 정의
rf = RandomForestClassifier(
    n_estimators=500,
    class_weight={0:1, 1:1, 2:1, 3:5},
    n_jobs=-1,
    random_state=42
)
svm = SVC(kernel='rbf', probability=True, random_state=42)
hybrid = VotingClassifier(
    estimators=[('rf', rf), ('svm', svm)],
    voting='soft',
    weights=[2, 1],
    n_jobs=-1
)
iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)

models = [
    ('Random Forest', rf, True),
    ('SVM', svm, True),
    ('Hybrid_RF_SVM', hybrid, True),
    ('Isolation Forest', iso_forest, False)
]

for name, model, is_supervised in models:
    print(f"\n>>> [{name}] 학습 시작...")
    start = time.time()

    if is_supervised:
        model.fit(X_train_res, y_train_res)
    else:
        model.fit(X_train_res)

    print(f">>> [{name}] 완료! (소요시간: {time.time()-start:.2f}초)")
    evaluate_model(model, X_test_scaled, y_test, name)
    joblib.dump(model, f'model_{name}.pkl')

joblib.dump(scaler, 'scaler.pkl')
print("\n[전체 완료] 모든 모델 학습 및 평가 완료!")
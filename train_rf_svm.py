# train_models.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

print("1. 데이터 로딩 중... (features.csv)")
df = pd.read_csv("features.csv")

X = df.drop('Label', axis=1)
y = df['Label']

# [중요] stratify=y 를 사용하여 소수 클래스(SQL Injection 등)가 
# 훈련셋과 테스트셋에 골고루 섞이게 합니다.
print("2. 데이터 분할 중 (Train 80% / Test 20%)")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1. Random Forest 학습
print("\n3. Random Forest 모델 학습 시작 (n_jobs=-1 사용)")
# train_rf_svm.py 내용 중 RF 선언 부분 수정
rf_model = RandomForestClassifier(
    n_estimators=200,          # 나무 개수를 늘려 좀 더 정교하게 학습
    max_depth=20,              # 과적합 방지를 위해 깊이 제한 (선택 사항)
    class_weight='balanced_subsample', # 소수 클래스에 더 강력한 가중치 부여
    n_jobs=-1, 
    random_state=42
)

rf_model.fit(X_train, y_train)
# 2. Linear SVM 학습
print("4. Linear SVM 모델 학습 시작 (Large scale용)")
# 대용량 데이터셋이므로 전통적인 SVC 대신 LinearSVC를 사용합니다.
svm_model = LinearSVC(dual=False, random_state=42) 
svm_model.fit(X_train, y_train)

# 결과 저장
print("\n5. 학습 결과 저장 중...")
joblib.dump(rf_model, "model_rf.pkl")
joblib.dump(svm_model, "model_svm.pkl")

# 평가 함수
def evaluate_model(model, name, X_test, y_test):
    print(f"\n--- {name} 성능 리포트 ---")
    y_pred = model.predict(X_test)
    # Target Names 매핑 (0~5)
    target_names = ['BENIGN', 'DDoS', 'PortScan', 'Web-BruteForce', 'Web-SQL', 'Web-XSS']
    print(classification_report(y_test, y_pred, target_names=target_names))

evaluate_model(rf_model, "Random Forest", X_test, y_test)
evaluate_model(svm_model, "Linear SVM", X_test, y_test)

print("\n✅ 모든 모델 학습 및 저장이 완료되었습니다.")

"""실습 05 심화: 같은 표준화 공간에서 K=2·3·4를 비교하고 새 요청에 같은 규칙을 씁니다."""
from pathlib import Path
import pandas as pd
import numpy as np
# 현재 실행 중인 파이썬 파일 위치
HERE = Path(__file__).resolve().parent
# 원본 데이터 로드
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# 학습/검증 분할 정보 로드
manifest = pd.read_csv(HERE / "data/split_manifest.csv")
# train 데이터의 request_id 목록
train_ids = manifest.loc[manifest["split"] == "train"].sort_values("position")["request_id"].tolist()
# valid 데이터의 request_id 목록
valid_ids = manifest.loc[manifest["split"] == "valid"].sort_values("position")["request_id"].tolist()
# train / valid 데이터 생성
train, valid = raw.loc[train_ids].copy(), raw.loc[valid_ids].copy()
# 사용할 입력 변수
FEATURES = ["input_chars", "allowed_tools", "task_type", "priority"]
# 수치형 변수와 범주형 변수 분리해서 선언
NUM, CAT = FEATURES[:2], FEATURES[2:]
# 입력 데이터 분리
X_train, X_valid = train[FEATURES].copy(), valid[FEATURES].copy()
# 목표 변수
y_train, y_valid = train["failed"], valid["failed"]
# 출력 폴더 생성
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
rows = []
# 같은 전처리 조건에서 K를 바꿔가며 비교
for k in [2, 3, 4]:
    # 결측치 대체 → 표준화 → KMeans 군집화
    model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),
                      ("cluster", KMeans(n_clusters=k, n_init=10, random_state=42))])
    # TODO H05: 현재 K의 model을 train 수치2열로 학습하세요. 비교할 때 전처리와 입력은 고정합니다.
    # pass
    # 수치형 변수(input_chars, allowed_tools)만 사용하여 학습
    model.fit(train[NUM])
    # TODO 끝
    # 군집화 이전 단계(결측치 처리 + 표준화) 결과
    z = model[:-1].transform(X_train[NUM])
    # 각 샘플이 배정된 군집 번호
    labels = model["cluster"].labels_
    # 군집 평가 지표 저장
    rows.append({"K": k, "inertia": model["cluster"].inertia_, "silhouette": silhouette_score(z, labels)})
    # K=3 모델은 이후 프로파일링 및 예측에 사용
    if k == 3:
        selected = model
# K별 평가 결과 저장
result = pd.DataFrame(rows)
result.to_csv(OUT / "cluster_scores.csv", index=False, encoding="utf-8-sig")
# 표준화 공간의 중심점을 원래 스케일로 복원
centers = selected["scaler"].inverse_transform(selected["cluster"].cluster_centers_)
# 군집 프로파일 생성
profile = pd.DataFrame(centers, columns=NUM)
# 군집 번호 추가
profile.insert(0, "cluster", range(3))
# 군집별 데이터 개수
profile["n"] = np.bincount(selected["cluster"].labels_, minlength=3)
# 군집 프로파일 저장
profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
# 새 요청 데이터
new = pd.DataFrame({"input_chars": [500, 2000], "allowed_tools": [1, 4]})
# 평가 결과 출력
print(result.round(4).to_string(index=False))
# 군집 특성 출력
print(profile.round(2).to_string(index=False))
# 새 요청을 K=3 군집 규칙으로 예측
print("새 요청의 K=3 군집:", selected.predict(new).tolist())
# inertia 의미 안내
print("inertia는 표준화 공간의 제곱거리 합입니다. F1과 평균내지 않습니다.")

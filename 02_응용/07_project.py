"""실습 07~08 응용: 06번 계획을 읽어 개발80으로 새 Pipeline을 학습하고 test20을 평가합니다."""
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# 스크립트가 위치한 디렉토리의 절대 경로 설정
HERE = Path(__file__).resolve().parent
# 결과물 저장 경로 지정
OUT = HERE / "outputs"
# 06번 단계에서 생성된 최적 설정 파일(selection.json)을 로드하여 딕셔너리로 변환
plan = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
# 원본 데이터셋을 불러온 뒤 request_id를 행 인덱스로 지정
raw = pd.read_csv(HERE / "data/requests.csv").set_index("request_id")
# selection.json에 정의된 분할 ID 정보를 가져옴
ids = plan["split_ids"]
# 학습/검증 데이터(개발용 80%)와 테스트 데이터(평가용 20%)를 각각 분리
develop = raw.loc[ids["train"] + ids["valid"]]
test = raw.loc[ids["test"]]
# 지정된 특성(features) 컬럼들만 복사하여 X_develop과 X_test 데이터프레임 구성
X_develop, X_test = develop[plan["features"]].copy(), test[plan["features"]].copy()
# 수치형 특성과 범주형 특성으로 분리하여 변수 설정
NUM, CAT = plan["features"][:2], plan["features"][2:]
# 제공: 같은 행별 공식을 새 입력에도 적용합니다. 학습 통계는 아래 fit(develop)에서만 계산됩니다.
# 파생 변수 생성 여부(derived)가 참인 경우, 자연로그 변환 및 슬롯당 글자 수 등 추가 특성을 수치형 데이터에 반영
if plan["derived"]:
    for part in (X_develop, X_test):
        part["log_input_chars"] = np.log1p(part["input_chars"])
        part["chars_per_slot"] = part["input_chars"] / (1.0 + part["allowed_tools"])
    NUM = NUM + ["log_input_chars", "chars_per_slot"]
# 수치형과 범주형 컬럼 각각에 대해 결측치 처리, 스케일링, 원핫 인코딩을 수행할 ColumnTransformer 정의
pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)), ("scaler", StandardScaler())]), NUM),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
                      ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), CAT),
])
# 전처리 파이프라인과 로지스틱 회귀 모델(Logistic Regression)을 결합하여 전체 파이프라인 구성
model = Pipeline([("pre", pre), ("model", LogisticRegression(C=plan["C"], max_iter=plan["max_iter"]))])
# TODO A07a: 새 Pipeline을 X_develop·failed로 학습하세요. 기존 train 통계를 재사용하지 않고 개발80에서 새로 fit합니다.
# pass
# 개발 데이터셋의 특성과 타겟값(failed)을 사용하여 새로운 파이프라인 모델을 처음부터 다시 학습
model.fit(X_develop, develop["failed"])
# TODO 끝
# 모델이 학습한 클래스 목록에서 양성(실패, 1) 레이블의 인덱스 위치를 확인
positive_column = model.classes_.tolist().index(1)
# TODO A07b: 확정 모델로 test의 실패 확률을 구하세요. test는 fit하지 않습니다.
# 테스트 데이터셋을 학습이 완료된 모델에 입력하여 예측 확률을 구한 뒤, 양성 클래스(1)에 해당하는 열만 추출
probability = model.predict_proba(X_test)[:, positive_column]
# TODO 끝
# 설정된 임계값(threshold)과 비교하여 0(정상) 또는 1(실패)의 이진 분류 예측값 생성
prediction = (probability > plan["threshold"]).astype(int)
# 혼동 행렬을 통해 참음성(TN), 거짓양성(FP), 거짓음성(FN), 참양성(TP) 개수 산출
tn, fp, fn, tp = confusion_matrix(test["failed"], prediction, labels=[0, 1]).ravel()
# 모델의 최종 평가 지표들을 딕셔너리 형태로 취합
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1_score(test["failed"], prediction, zero_division=0),
           "ROC_AUC": roc_auc_score(test["failed"], probability),
           "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp), "actual_failed": int(test["failed"].sum())}
# request_id, 실제값, 예측 실패 확률, 이진 예측값을 포함하는 결과 데이터프레임 생성
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
# 결과 예측 데이터와 평가지표를 outputs 폴더 내에 각각 CSV 및 JSON 파일로 저장 (한글 깨짐 방지 utf-8-sig 인코딩 적용)
result.to_csv(OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
(OUT / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
# 확정된 모델 정보, 평가지표, 예측 오류 사례를 콘솔 창에 출력
print("확정 모델:", plan["selected"], "/ 학습80·최종20")
print(pd.DataFrame([metrics]).round(4).to_string(index=False))
print("오류 사례")
print(result.loc[result["actual"] != result["predicted"]].head(3).round(3).to_string(index=False))
print("저장: outputs/test_predictions.csv, test_metrics.json")
print("08번에서는 이 설정·원본을 그대로 재실행합니다. test를 보고 후보를 다시 고르지 않습니다.")

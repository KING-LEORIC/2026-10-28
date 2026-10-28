"""실습 07~08: 06번의 설정을 그대로 읽어 개발 16건으로 학습하고 test 4건을 최종 평가합니다."""
from pathlib import Path
import pandas as pd
import numpy as np
import json
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
# 현재 파이썬 스크립트가 실행되는 디렉토리의 절대 경로를 안전하게 확보
HERE = Path(__file__).resolve().parent
# 제공: 06번을 먼저 완료하세요. 이 프로그램은 CV·설정 선택을 다시 하지 않습니다.
# 06번 단계에서 최적화되어 저장된 설정 파일(selection.json)을 읽어 딕셔너리 형태로 변환
plan = json.loads((HERE / "selection.json").read_text(encoding="utf-8"))
# 전체 원본 데이터셋(requests.csv)을 불러온 뒤, 각 요청의 고유 식별자인 request_id를 데이터프레임의 행 인덱스로 지정
raw = pd.read_csv(HERE / "requests.csv").set_index("request_id")
# selection.json에 기록된 학습용 ID(train_ids)와 검증용 ID(valid_ids)를 결합하여 개발 데이터셋(16건) 구성
develop = raw.loc[plan["train_ids"] + plan["valid_ids"]].copy()
# selection.json에 기록된 테스트용 ID(test_ids)만 추출하여 최종 평가를 위한 독립된 테스트 데이터셋(4건) 분리
test = raw.loc[plan["test_ids"]].copy()
# 제공: 행별 공식은 새 요청에도 동일하게 적용합니다. test로 통계나 설정을 학습하지 않습니다.
# 개발 및 테스트 데이터 모두에 대해 입력 문자열 길이(input_chars)의 분포 편향을 완화하고 모델 학습 안정성을 높이기 위해 자연로그 변환(log1p) 수행
for table in (develop, test):
    table["log_input_chars"] = np.log1p(table["input_chars"])
# selection.json에 저장된 최적 하이퍼파라미터(max_depth, random_state)를 바탕으로 의사결정나무(Decision Tree) 모델 객체 초기화
model = DecisionTreeClassifier(max_depth=plan["max_depth"], random_state=plan["random_state"])
# TODO B07a: 개발 자료 16건의 선택된 입력과 failed로 학습하세요. test는 넣지 않습니다.
# pass
# selection.json의 features에 정의된 특성 열들만 골라내고 타겟값인 failed(실패 여부)를 매칭하여 모델 학습 진행
# - 테스트 데이터는 엄격히 배제
model.fit(develop[plan["features"]], develop["failed"])
# TODO 끝
# 0·1 두 클래스가 있는 정상 자료입니다. classes_에서 양성 1의 위치를 확인합니다.
# 학습된 모델이 인식하는 클래스 목록에서 양성(실패, 1)에 해당하는 클래스의 인덱스 위치를 탐색
positive_column = model.classes_.tolist().index(1)
# TODO B07b: test의 실패 확률을 구하세요. predict_proba의 positive_column 열만 선택합니다.
# 테스트 데이터의 특성값을 모델에 입력해 각 클래스별 예측 확률을 산출한 뒤, 양성 클래스(1)에 해당하는 열만 추출
probability = model.predict_proba(test[plan["features"]])[:, positive_column]
# TODO 끝
# 산출된 실패 확률이 selection.json의 threshold(임계값)를 초과하는지 비교하여 0(정상) 또는 1(실패)의 이진 분류 예측값 생성
prediction = (probability > plan["threshold"]).astype(int)
# TODO B07c: F1에는 0/1 예측값, AUC에는 실패 확률을 넣으세요.
# 실제 정답(test["failed"])과 이진 예측값(prediction)을 비교해 F1 Score를 구하고, 정답과 연속형 예측 확률(probability)을 비교해 ROC AUC 산출
f1 = f1_score(test["failed"], prediction, zero_division = 0)
auc = roc_auc_score(test["failed"], probability)
# TODO 끝
# 모델의 최종 성능 평가 지표(정확도, 정밀도, 재현율, F1, AUC 등)를 딕셔너리 형태로 취합
metrics = {"n": len(test), "Accuracy": accuracy_score(test["failed"], prediction),
           "Precision": precision_score(test["failed"], prediction, zero_division=0),
           "Recall": recall_score(test["failed"], prediction, zero_division=0),
           "F1": f1, "ROC_AUC": auc, "TP": int(((test["failed"] == 1) & (prediction == 1)).sum()),
           "actual_failed": int(test["failed"].sum()), "threshold": plan["threshold"]}
# request_id, 실제값, 예측 실패 확률, 이진 예측값을 포함하는 결과 데이터프레임 생성
result = pd.DataFrame({"request_id": test.index, "actual": test["failed"].to_numpy(),
                       "probability_failed": probability, "predicted": prediction})
# 계산된 예측 결과와 평가지표를 각각 CSV 및 JSON 파일로 저장 (한글 깨짐 방지를 위해 utf-8-sig 인코딩 적용)
result.to_csv(HERE / "test_predictions.csv", index=False, encoding="utf-8-sig")
(HERE / "test_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
# 개발/평가 데이터 크기, 예측 결과 테이블, 혼동행렬, 주요 평가지표를 콘솔에 출력하여 확인
print("개발/최종평가:", len(develop), len(test))
print(result.round(3).to_string(index=False))
print("혼동행렬:", confusion_matrix(test["failed"], prediction, labels=[0, 1]).tolist())
print(f"Accuracy={metrics['Accuracy']:.3f}, Precision={metrics['Precision']:.3f}, Recall={metrics['Recall']:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
print("저장: test_predictions.csv, test_metrics.json")
print("08번은 이 설정 그대로 재실행합니다. test를 보고 06번 설정을 다시 고르지 않습니다.")

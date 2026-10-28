"""실습 01 심화: 지표와 업무 비용을 함께 읽습니다. 기존 응용 수준의 짧은 비교입니다."""
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# 결과 저장 폴더 생성
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
# 실제 정답 데이터
# 0=성공, 1=실패
# 성공 14건, 실패 6건 예시
y = [0]*14 + [1]*6
# 비교할 예측 결과
# - example : 일부 실패를 탐지하는 모델
# - all_success : 모든 건을 성공(0)으로 예측하는 모델
predictions = {"example": [0]*10 + [1]*4 + [0]*4 + [1]*2, "all_success": [0]*20}
rows = []
# 모델별 평가
for name, pred in predictions.items():
    # 혼동행렬로 TN, FP, FN, TP 계산
    # TN: 성공을 성공으로 맞춤
    # FP: 성공인데 실패라고 판단(오탐)
    # FN: 실패인데 성공이라고 판단(미탐)
    # TP: 실패를 실패로 맞춤
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # TODO H01: FP 비용1, FN 비용5라는 가정으로 총비용을 계산하세요.
    
    # 정확도(Accuracy)는 모든 오답을 동일하게 취급하지만, 실제 업무에서는 오답의 종류에 따라 손실이 다를 수 있기 때문에
    # 여기서는:
    # - FP(False Positive): 비용 1
    # - FN(False Negative): 비용 5
    # 로 가정
    # 즉, 실제 실패를 놓치는(FN) 상황이 더 큰 문제라는 의미.
    # 비용을 함께 계산하면 단순 정확도뿐 아니라 업무 관점에서 더 유리한 모델을 평가할 수 있기 때문.
    cost =  fp * 1 + fn * 5
    # TODO 끝
    # 일반적인 분류 성능 지표 계산
    # - assumed_cost : 업무 관점의 비용 지표
    rows.append({"model": name, "Accuracy": accuracy_score(y, pred),
                 "Precision": precision_score(y, pred, zero_division=0),
                 "Recall": recall_score(y, pred, zero_division=0),
                 "F1": f1_score(y, pred, zero_division=0), "FP": fp, "FN": fn,
                 "assumed_cost": cost})
# 결과를 표 형태로 정리
result = pd.DataFrame(rows)
# CSV 저장
result.to_csv(OUT / "metric_comparison.csv", index=False, encoding="utf-8-sig")
# 소수점 3자리까지 출력
print(result.round(3).to_string(index=False))
# 실습 의도 설명
print("가정: 오탐 1회 비용1, 실제 실패를 놓친 비용5. 비용 단위는 임의 점수입니다.")

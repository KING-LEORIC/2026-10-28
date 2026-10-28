"""실습 02 심화: ROC 좌표와 두 임계값 표를 별도로 기록합니다."""
# 1. ROC 곡선 좌표(FPR, TPR, threshold)를 저장
# 2. 임계값(0.5, 0.3)에 따라 예측 결과가 어떻게 달라지는지 비교
# 3. AUC, F1, 비용(cost)을 계산
# ※ 실제 모델을 재학습하는 것이 아니라, 이미 존재하는 score에 대해 임계값만 바꿔보는 예제
from pathlib import Path
import numpy as np
import pandas as pd
# ROC, 혼동행렬, F1, AUC 계산 함수
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
# 결과 저장 폴더 생성
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
# 실제 정답(label)
# 0 = 음성(성공)
# 1 = 양성(실패)
y = np.array([0, 0, 1, 1])
# 모델이 예측한 양성(실패) 확률
score = np.array([0.10, 0.40, 0.35, 0.80])
# ROC 곡선 계산
# - FPR(False Positive Rate)
# - TPR(True Positive Rate)
# - threshold
fpr, tpr, thresholds = roc_curve(y, score)
# ROC 좌표 저장
pd.DataFrame({"FPR": fpr, "TPR": tpr, "threshold": thresholds}).to_csv(OUT / "roc_points.csv", index=False, encoding="utf-8-sig")
# 임계값별 결과 저장용 리스트
rows = []
# 임계값 비교 실험
for threshold in [0.5, 0.3]:
    # TODO H02: 현재 임계값보다 점수가 클 때만 양성1로 바꾸세요.
    # 처음에는 전부 음성(0)으로 예측
    pred = np.zeros(len(y), dtype=int)
    # 현재 임계값보다 큰 점수만 양성(1) 처리
    # score == threshold인 경우는 0 유지
    pred[score > threshold] = 1
    # TODO 끝
    # 혼동행렬 계산
    #           예측
    #          0    1
    # 실제 0   TN   FP
    # 실제 1   FN   TP
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    # 결과 저장
    # - threshold : 임계값
    # - fp : False Positive
    # - fn : False Negative
    # - f1_score(y, pred) : Precision과 Recall을 함께 고려한 점수
    # - 비용 가정 : FP 비용 = 1, FN 비용 = 5
    # - AUC는 score 자체를 평가하므로 threshold를 바꿔도 동일
    rows.append({"threshold": threshold, "FP": fp, "FN": fn, "F1": f1_score(y, pred),
                 "cost": fp+5*fn, "AUC": roc_auc_score(y, score)})
# 결과를 표 형태로 변환
result = pd.DataFrame(rows)
# CSV 저장
result.to_csv(OUT / "threshold_comparison.csv", index=False, encoding="utf-8-sig")
# 화면 출력
print(result.round(3).to_string(index=False))
# ROC 설명
print("ROC 첫 임계값 inf는 어떤 값도 양성으로 정하지 않는 시작점입니다. 입력 오류가 아닙니다.")
# 실습 목적 설명
print("이 원리 예제는 최종 분류기의 임계값을 변경하는 실험이 아닙니다.")

import os
import cv2
import dlib
from gaze_tracking import GazeTracking
from l2cs import Pipeline, render
import torch
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 현재 작업 디렉토리 설정 (필요 시 수정)
CWD = Path.cwd()

# GazeTracking 초기화
gaze = GazeTracking()

# dlib의 얼굴 인식기와 랜드마크 불러오기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/jihwankim/Desktop/대학원 논문/Eye-Tracking/model_final/shape_predictor_68_face_landmarks.dat")

# 가중치 파일 경로 설정 (경로를 raw 문자열로 사용)
weights_path = '/Users/jihwankim/Desktop/대학원 논문/Eye-Tracking/model_final/L2CSNet_gaze360.pkl'

# L2CS 파이프라인 설정
gaze_pipeline = Pipeline(
    weights=weights_path,
    arch='ResNet50',
    device=torch.device('cpu')
)

# 테스트 데이터셋 디렉토리 설정
base_dir = Path('/Users/jihwankim/Desktop/대학원 논문/Eye-Tracking/model_final/datatimes')

# 성능 지표를 위한 리스트 초기화
true_labels = []
predicted_labels_gaze = []
predicted_labels_l2cs = []

# 저장된 모든 비디오 파일 탐색
for folder in base_dir.iterdir():
    if folder.is_dir():
        for file in folder.iterdir():
            if file.suffix == '.mp4':
                cap = cv2.VideoCapture(str(file))

                if not cap.isOpened():
                    print(f"파일을 열 수 없습니다: {file}")
                    continue

                while True:
                    # 프레임 읽기
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 실제 레이블 설정 (테스트 데이터셋에 따라 다름, 예시로 'center' 설정)
                    true_labels.append('center')  # 실제 값으로 교체 필요

                    # GazeTracking 분석
                    gaze.refresh(frame)
                    gaze_result = "center"
                    if gaze.is_blinking():
                        gaze_result = "blinking"
                    elif gaze.is_right():
                        gaze_result = "right"
                    elif gaze.is_left():
                        gaze_result = "left"
                    elif gaze.is_center():
                        gaze_result = "center"

                    predicted_labels_gaze.append(gaze_result)

                    # L2CS Gaze Estimation
                    results = gaze_pipeline.step(frame)
                    # 'pitch'와 'yaw' 값을 사용하여 시선 방향 결정
                    pitch = results.pitch
                    yaw = results.yaw

                    # 예시로 pitch와 yaw 값을 기반으로 시선 방향을 설정
                    if abs(pitch) < 10 and abs(yaw) < 10:
                        l2cs_result = 'center'
                    elif yaw > 10:
                        l2cs_result = 'right'
                    elif yaw < -10:
                        l2cs_result = 'left'
                    elif pitch > 10:
                        l2cs_result = 'up'
                    else:
                        l2cs_result = 'down'

                    predicted_labels_l2cs.append(l2cs_result)

                # 자원 해제
                cap.release()

# 성능 지표 계산
gaze_accuracy = accuracy_score(true_labels, predicted_labels_gaze)
gaze_precision = precision_score(true_labels, predicted_labels_gaze, average='weighted')
gaze_recall = recall_score(true_labels, predicted_labels_gaze, average='weighted')
gaze_f1 = f1_score(true_labels, predicted_labels_gaze, average='weighted')

l2cs_accuracy = accuracy_score(true_labels, predicted_labels_l2cs)
l2cs_precision = precision_score(true_labels, predicted_labels_l2cs, average='weighted')
l2cs_recall = recall_score(true_labels, predicted_labels_l2cs, average='weighted')
l2cs_f1 = f1_score(true_labels, predicted_labels_l2cs, average='weighted')


print("GazeTracking 성능 지표")
print(f"정확도: {gaze_accuracy}")
print(f"정밀도: {gaze_precision}")
print(f"재현율: {gaze_recall}")
print(f"F1-score: {gaze_f1}")

print("L2CS 성능 지표")
print(f"정확도: {l2cs_accuracy}")
print(f"정밀도: {l2cs_precision}")
print(f"재현율: {l2cs_recall}")
print(f"F1-score: {l2cs_f1}")

cv2.destroyAllWindows()

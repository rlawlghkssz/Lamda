import os
import cv2
import dlib
from gaze_tracking import GazeTracking
from l2cs import Pipeline, render
import torch
from pathlib import Path
from datetime import datetime
import pandas as pd

# 현재 작업 디렉토리 설정 (필요 시 수정)
CWD = Path.cwd()

# 사용자 이름 입력받기
user_name = input("사용자 이름을 입력하세요: ")

# GazeTracking 초기화
gaze = GazeTracking()

# dlib의 얼굴 인식기와 랜드마크 불러오기
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("/Users/jihwankim/Desktop/model_final/shape_predictor_68_face_landmarks.dat")

# 가중치 파일 경로 설정 (경로를 raw 문자열로 사용)
weights_path = '/Users/jihwankim/Desktop/model_final/L2CSNet_gaze360.pkl'

# L2CS 파이프라인 설정
gaze_pipeline = Pipeline(
    weights=weights_path,
    arch='ResNet50',
    device=torch.device('cpu')
)

# 웹캠 인덱스 설정 (기본 웹캠은 0번 인덱스)
cam = 0
cap = cv2.VideoCapture(cam)

if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit()

# 저장 디렉토리 설정
base_dir = Path('/Users/jihwankim/Desktop/model_final/datatimes')
current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
output_dir = base_dir / current_time
output_dir.mkdir(parents=True, exist_ok=True)

# 비디오 저장을 위한 설정
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 코덱 설정
fps = 20.0  # 프레임 속도 설정
frame_width = int(cap.get(3))  # 프레임 너비 설정
frame_height = int(cap.get(4))  # 프레임 높이 설정
output_path = output_dir / 'output.mp4'
out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

# 좌표값 저장을 위한 데이터프레임 초기화
data = {
    "User": [],
    "Frame": [],
    "Left_Pupil_X": [],
    "Left_Pupil_Y": [],
    "Right_Pupil_X": [],
    "Right_Pupil_Y": [],
    "Direction": []
}

frame_count = 0

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽어오지 못했습니다.")
        break

    # 얼굴 인식 및 랜드마크 탐지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 왼쪽 눈 (landmark index 36-41)
        for n in range(36, 42):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 반지름을 1로 설정

        # 오른쪽 눈 (landmark index 42-47)
        for n in range(42, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # 반지름을 1로 설정

    # GazeTracking 분석을 위해 프레임을 보냅니다.
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    # L2CS Gaze Estimation
    results = gaze_pipeline.step(frame)
    frame = render(frame, results)

    # 좌표값 및 방향 저장
    data["User"].append(user_name)
    data["Frame"].append(frame_count)
    data["Left_Pupil_X"].append(left_pupil[0] if left_pupil else None)
    data["Left_Pupil_Y"].append(left_pupil[1] if left_pupil else None)
    data["Right_Pupil_X"].append(right_pupil[0] if right_pupil else None)
    data["Right_Pupil_Y"].append(right_pupil[1] if right_pupil else None)
    data["Direction"].append(text)
    
    frame_count += 1

    # 결과 프레임 비디오 파일에 기록
    out.write(frame)

    # 결과 프레임 보여주기
    cv2.imshow('Gaze Estimation & Eye Tracker', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
out.release()
cv2.destroyAllWindows()

# CSV 파일로 저장
df = pd.DataFrame(data)
csv_path = output_dir / 'coordinates.csv'
df.to_csv(csv_path, index=False)


#기본 모델만
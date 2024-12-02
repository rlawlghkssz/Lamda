import cv2
from pathlib import Path

# 비디오 파일 경로 설정
video_path = Path("/Users/jihwankim/Desktop/model_final/datatimes/20240823_165306/output.mp4")

# 프레임을 저장할 디렉토리 설정
output_dir = video_path.parent / 'frames'
output_dir.mkdir(parents=True, exist_ok=True)

# 비디오 파일 열기
cap = cv2.VideoCapture(str(video_path))

frame_count = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    # 프레임 저장
    frame_filename = output_dir / f'frame_{frame_count:04d}.jpg'
    cv2.imwrite(str(frame_filename), frame)

    frame_count += 1

cap.release()
print(f'총 {frame_count}개의 프레임이 {output_dir}에 저장되었습니다.')

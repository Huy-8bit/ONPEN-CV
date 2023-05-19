import face_recognition
import cv2
import os
import numpy as np

video_capture = cv2.VideoCapture(0)

# Tạo thư mục 'face' nếu chưa tồn tại
os.makedirs("face", exist_ok=True)

known_face_encodings = []
known_face_files = []

# Load và mã hóa các khuôn mặt đã biết từ thư mục 'face'
for file in os.listdir("face"):
    image = face_recognition.load_image_file(f"face/{file}")
    face_encodings = face_recognition.face_encodings(image)

    if len(face_encodings) > 0:
        known_face_encodings.append(face_encodings[0])
        known_face_files.append(file)

# Biến đếm frame
frame_count = 0

while True:
    ret, frame = video_capture.read()

    # Chỉ xử lý mỗi n khung hình, ở đây ta đặt n=1
    if frame_count % 1 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        for (top, right, bottom, left), face_encoding in zip(
            face_locations, face_encodings
        ):
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )

            if not any(matches):
                top, right, bottom, left = [
                    coordinate * 4 for coordinate in (top, right, bottom, left)
                ]
                face_image = frame[top:bottom, left:right]
                cv2.imwrite(f"face/face_{frame_count}.jpg", face_image)

                known_face_encodings.append(face_encoding)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 0)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

video_capture.release()
cv2.destroyAllWindows()

import face_recognition
import cv2
import os
import numpy as np
import threading
import tkinter as tk
from tkinter import simpledialog

# Tạo cửa sổ Tkinter để nhập tên
root = tk.Tk()
root.withdraw()

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
        known_face_files.append(
            os.path.splitext(file)[0]
        )  # Lấy tên file (loại bỏ phần mở rộng)

# Biến đếm frame
frame_count = 0

# Biến đồng bộ hóa cho việc sửa tên
lock = threading.Lock()


def modify_name(matches, face_locations):
    for i, match in enumerate(matches):
        person_index = -1
        person_name = ""

        if True in match:
            person_index = match.index(True)

        if person_index >= 0:
            # Hiển thị hộp thoại nhập tên để sửa tên
            person_name = simpledialog.askstring(
                "Sửa tên", f"Sửa tên của người thứ {i+1}:", parent=root
            )

        if person_name:
            with lock:
                # Sửa tên trong danh sách các khuôn mặt đã biết
                known_face_files[person_index] = person_name

    for (top, right, bottom, left), matches in zip(face_locations, matches):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Hiển thị tên khuôn mặt
        with lock:
            face_encoding = face_recognition.face_encodings(
                frame, [(top, right, bottom, left)]
            )[0]
            name = "Unknown"

            if True in matches:
                match_index = matches.index(True)
                name = known_face_files[match_index]

        cv2.putText(
            frame,
            name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            1,
        )


while True:
    ret, frame = video_capture.read()

    # Chỉ xử lý mỗi n khung hình, ở đây ta đặt n=1
    if frame_count % 1 == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)

        matches_list = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(
                known_face_encodings, face_encoding
            )
            matches_list.append(matches)

        # Sử dụng đa luồng (multithreading) để sửa tên
        modify_thread = threading.Thread(
            target=modify_name, args=(matches_list, face_locations), daemon=True
        )
        modify_thread.start()

    for top, right, bottom, left in face_locations:
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Vẽ khung đỏ xung quanh khuôn mặt
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    frame_count += 1

video_capture.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import os
import time

# Khởi tạo bộ nhớ đệm chuyển động
motion_history = None

# Khởi tạo video capture từ webcam
cap = cv2.VideoCapture(0)

# Index to save images with different names
image_index = 0

while True:
    ret, frame = cap.read()

    # Chuyển đổi hình ảnh sang mức xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Nếu lịch sử chuyển động vẫn trống, cập nhật nó
    if motion_history is None:
        motion_history = gray
        continue

    # Tính toán khác biệt tuyệt đối giữa hình ảnh hiện tại và hình ảnh lịch sử
    frame_delta = cv2.absdiff(motion_history, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]

    # Thực hiện một loạt các mở rộng và co lại để loại bỏ các vùng nhỏ
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Duyệt qua các đường viền
    for contour in contours:
        # Nếu diện tích đường viền nhỏ hơn một ngưỡng, bỏ qua nó
        if cv2.contourArea(contour) < 10000:
            continue

        # Tính hình chữ nhật bao quanh đường viền và vẽ nó trên khung hình
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Save the region of motion to a file
        object_image = frame[y : y + h, x : x + w]
        cv2.imwrite(f"photos/moving_object_{image_index}.jpg", object_image)
        image_index += 1

    # Hiển thị khung hình
    cv2.imshow("Motion Tracker", frame)

    # Cập nhật lịch sử chuyển động
    motion_history = gray

    # Nếu người dùng nhấn 'q', thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
import cv2
import openface

# Khởi tạo bộ nhận dạng khuôn mặt OpenFace
align = openface.AlignDlib()
net = openface.TorchNeuralNet(
    "./models/openface/nn4.small2.v1.t7", imgDim=96, cuda=False
)

# Khởi tạo webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Đọc frame từ webcam
    ret, frame = video_capture.read()

    # Đảm bảo frame đọc thành công
    if not ret:
        break

    # Chuyển đổi frame sang grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện khuôn mặt trong frame
    face_rects = align.getAllFaceBoundingBoxes(gray)

    # Duyệt qua từng khuôn mặt
    for face_rect in face_rects:
        # Trích xuất các đặc trưng của khuôn mặt
        aligned_face = align.align(
            96, gray, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE
        )
        rep = net.forward(aligned_face)

        # Hiển thị khuôn mặt và đặc trưng trên frame
        cv2.rectangle(
            frame,
            (face_rect.left(), face_rect.top()),
            (face_rect.right(), face_rect.bottom()),
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            "Face",
            (face_rect.left(), face_rect.top() - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            frame,
            str(rep),
            (face_rect.left(), face_rect.bottom() + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

    # Hiển thị frame kết quả
    cv2.imshow("Video", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng tài nguyên
video_capture.release()
cv2.destroyAllWindows()

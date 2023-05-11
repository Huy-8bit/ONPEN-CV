import cv2
import face_recognition

# Load bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load ảnh mẫu và tạo mã nhận dạng
image_sample = face_recognition.load_image_file("face.jpg")
sample_encoding = face_recognition.face_encodings(image_sample)[0]

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có được mở hay không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Vòng lặp để đọc từng khung hình từ camera và nhận diện khuôn mặt
while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    # Nếu không đọc được khung hình thì thoát vòng lặp
    if not ret:
        break

    # Nhận diện khuôn mặt trong ảnh
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # So sánh khuôn mặt với mẫu
    for (top, right, bottom, left), face_encoding in zip(
        face_locations, face_encodings
    ):
        matches = face_recognition.compare_faces([sample_encoding], face_encoding)
        name = "Unknown"

        if True in matches:
            name = "Matched"

        # Vẽ hình chữ nhật xung quanh khuôn mặt và ghi tên
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(
            frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2
        )

    # Hiển thị khung hình
    cv2.imshow("Camera", frame)

    # Nhấn phím 'q' để thoát vòng lặp
    if cv2.waitKey(1) == ord("q"):
        break

# Giải phóng bộ nhớ và dừng camera
cap.release()
cv2.destroyAllWindows()

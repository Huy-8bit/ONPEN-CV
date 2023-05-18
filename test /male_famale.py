import cv2

# Load bộ phân loại khuôn mặt và khởi tạo camera
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
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

    # Chuyển khung hình sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Nhận diện khuôn mặt trong ảnh xám
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Vẽ hình chữ nhật xung quanh các khuôn mặt được nhận diện
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Tính toán độ dốc trung bình của khuôn mặt
        face_roi = gray[y : y + h, x : x + w]
        mean_gradient = cv2.mean(face_roi)[0]

        # Xác định giới tính của khuôn mặt dựa trên độ dốc trung bình
        gender = "Nam" if mean_gradient > 20 else "Nữ"

        # Hiển thị giới tính lên khung hình
        cv2.putText(
            frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2
        )

    # Hiển thị khung hình lên màn hình
    cv2.imshow("Camera", frame)

    # Nhấn phím 'q' để thoát vòng lặp
    if cv2.waitKey(1) == ord("q"):
        break

# Giải phóng bộ nhớ và dừng camera
cap.release()
cv2.destroyAllWindows()

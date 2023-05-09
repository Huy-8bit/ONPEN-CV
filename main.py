import cv2

# Khởi tạo bộ phân loại khuôn mặt
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Khởi tạo đối tượng VideoCapture để truy cập vào camera
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

    # Hiển thị khung hình kết quả
    cv2.imshow("Camera", frame)

    # Nhấn phím ESC để thoát khỏi vòng lặp
    if cv2.waitKey(1) == 27:
        break

# Giải phóng đối tượng VideoCapture và đóng tất cả các cửa sổ hiển thị
cap.release()
cv2.destroyAllWindows()

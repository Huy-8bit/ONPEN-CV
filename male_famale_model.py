import cv2

# Load bộ phân loại khuôn mặt và khởi tạo camera
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

# Load mô hình giới tính
gender_model = cv2.dnn.readNetFromCaffe(
    "deploy_gender.prototxt", "gender_net.caffemodel"
)
gender_list = ["Nam", "Nữ"]

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

        # Đưa khuôn mặt vào mô hình giới tính để nhận diện
        face_img = frame[y : y + h, x : x + w].copy()
        blob = cv2.dnn.blobFromImage(
            face_img,
            1.0,
            (227, 227),
            (78.4263377603, 87.7689143744, 114.895847746),
            swapRB=False,
        )
        gender_model.setInput(blob)
        gender_preds = gender_model.forward()
        gender = gender_list[gender_preds[0].argmax()]

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

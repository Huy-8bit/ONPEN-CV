import cv2

# Danh sách các file cascade XML và tên tương ứng
cascade_files = [
    # ("haarcascade_eye.xml", "Eye"),
    # ("haarcascade_eye_tree_eyeglasses.xml", "Eye with Eyeglasses"),
    # ("haarcascade_frontalcatface.xml", "Cat Face"),
    # ("haarcascade_frontalcatface_extended.xml", "Extended Cat Face"),
    # ("haarcascade_frontalface_alt.xml", "Frontal Face (alt)"),
    # ("haarcascade_frontalface_alt2.xml", "Frontal Face (alt2)"),
    # ("haarcascade_frontalface_alt_tree.xml", "Frontal Face (alt tree)"),
    # ("haarcascade_frontalface_default.xml", "Frontal Face (default)"),
    # ("haarcascade_fullbody.xml", "Full Body"),
    # ("haarcascade_lefteye_2splits.xml", "Left Eye (2 Splits)"),
    # ("haarcascade_license_plate_rus_16stages.xml", "License Plate (Russian)"),
    # ("haarcascade_lowerbody.xml", "Lower Body"),
    # ("haarcascade_profileface.xml", "Profile Face"),
    # ("haarcascade_righteye_2splits.xml", "Right Eye (2 Splits)"),
    # ("haarcascade_russian_plate_number.xml", "Russian Plate Number"),
    # ("haarcascade_smile.xml", "Smile"),
    # ("haarcascade_upperbody.xml", "Upper Body"),
]

# Khởi tạo đối tượng VideoCapture để đọc video từ camera
cap = cv2.VideoCapture(0)

# Kiểm tra xem camera có được mở thành công hay không
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Đọc khung hình đầu tiên từ camera
ret, frame = cap.read()

# Đảm bảo video đầu vào hợp lệ
if not ret:
    print("Không thể đọc khung hình từ camera")
    exit()


# Định nghĩa hàm phát hiện vật thể
def detect_objects(frame, cascade_file, object_name):
    # Tạo đối tượng CascadeClassifier từ file cascade
    cascade = cv2.CascadeClassifier(cascade_file)

    # Chuyển sang ảnh xám để tăng tốc độ xử lý
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Phát hiện vật thể sử dụng cascade
    objects = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
    )

    # Vẽ khung bao xung quanh các vật thể
    for x, y, w, h in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            object_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

    return frame


# Đọc các khung hình từ camera
while True:
    # Đọc khung hình từ camera
    ret, frame = cap.read()

    if not ret:
        print("Không thể đọc khung hình từ camera")
        break

    # Duyệt qua danh sách các file cascade và vẽ khung bao tương ứng
    for cascade_file, object_name in cascade_files:
        frame = detect_objects(frame, cascade_file, object_name)

    # Hiển thị khung hình kết quả
    cv2.imshow("Object Tracking", frame)

    # Kiểm tra sự kiện nhấn phím 'q' để thoát
    if cv2.waitKey(1) == ord("q"):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()

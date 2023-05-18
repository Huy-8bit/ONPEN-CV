import cv2
import os
import numpy as np


def get_images_and_labels(path):
    image_paths = [
        os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jpg")
    ]
    face_samples = []
    ids = []

    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_numpy = np.array(img, "uint8")
        id = int(os.path.split(image_path)[-1].split("_")[1].split(".")[0])
        face_samples.append(img_numpy)
        ids.append(id)

    return face_samples, ids


# Initialize the face cascade classifier
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Initialize VideoCapture object to access the camera
cap = cv2.VideoCapture(0)

# Check if the camera has been opened
if not cap.isOpened():
    print("Cannot open camera")
    exit()

frame_height = 720
frame_width = frame_height * 16 / 9
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Create 'face' directory if it does not exist
os.makedirs("faces", exist_ok=True)

# LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer with existing data
faces, ids = get_images_and_labels("face")

if len(faces) > 0:
    recognizer.train(faces, np.array(ids))
else:
    print("No faces for training. Please add face images in 'face' directory.")
    exit()

# Initialize image counter
image_counter = len(faces)

# Loop to read each frame from the camera and recognize faces
while True:
    # Read frame from camera
    ret, frame = cap.read()

    # If the frame cannot be read then break the loop
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw rectangle around the detected faces and recognize them
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Recognize the face
        id, confidence = recognizer.predict(gray[y : y + h, x : x + w])

        # If confidence is high, the face is recognized; otherwise, save the face to 'face' directory and retrain the recognizer
        if confidence < 100:
            cv2.putText(
                frame,
                f"ID {id}",
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
        else:
            face_img = frame[y : y + h, x : x + w]
            cv2.imwrite(f"face/face_{image_counter}.jpg", face_img)
            image_counter += 1

            # Retrain the recognizer with the new face
            faces, ids = get_images_and_labels("face")
            recognizer.train(faces, np.array(ids))

    # Show the resulting frame
    cv2.imshow("Camera", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord("q"):
        break

# Release the VideoCapture object and close all display windows
cap.release()
cv2.destroyAllWindows()

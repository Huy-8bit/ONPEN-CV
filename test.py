import cv2

print(cv2.__version__)
# Set up tracker
tracker = cv2.TrackerCSRT_create()

# Read video
video = cv2.VideoCapture(0)

# Exit if video not opened
if not video.isOpened():
    print("Could not open video")
    sys.exit()

# Read first frame
ok, frame = video.read()
if not ok:
    print("Cannot read video file")
    sys.exit()

# Let the user draw the initial bounding box
bbox = cv2.selectROI(frame, False)

# Initialize tracker with first frame and bounding box
ok = tracker.init(frame, bbox)

while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Draw bounding box
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (0, 0, 255))

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture when everything is done
video.release()
cv2.destroyAllWindows()

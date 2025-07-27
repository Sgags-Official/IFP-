import cv2
import math
import cvzone
from ultralytics import YOLO

# Initialize video capture
video_path = "Media/Potholes.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights
model = YOLO("Weights/best.pt")

# Define class names
class_names = ['Pothole']

while True:
    success, img = cap.read()
    if not success:  # Exit when video ends
        print("End of video or cannot read frame.")
        break

    # Perform detection
    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Convert to int
            w, h = x2 - x1, y2 - y1

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.4:  # Confidence threshold
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{class_names[cls]} {conf:.2f}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Show result
    cv2.imshow("Pothole Detection", img)

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

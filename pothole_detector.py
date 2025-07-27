import cv2
import math
import cvzone
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("Weights/best.pt")

# Class names
class_labels = ['Pothole']

# Load image
image_path = "Media/pothole_7.jpg"
img = cv2.imread(image_path)

# Predict
results = yolo_model(img)

# Draw results
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
        conf = float(box.conf[0])               # Confidence
        cls = int(box.cls[0])                   # Class index

        if conf > 0.3:
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), t=2)
            cvzone.putTextRect(img, f'{class_labels[cls]} {conf:.2f}', (x1, y1 - 10),
                               scale=0.8, thickness=1, colorR=(255, 0, 0))

# Show result
cv2.imshow("Pothole Detection", img)

# Wait until 'q' is pressed
while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

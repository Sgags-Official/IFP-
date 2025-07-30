import cv2
import math
import cvzone
import torch
from ultralytics import YOLO

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Initialize video capture
video_path = "Media/Potholes_3.mp4"
cap = cv2.VideoCapture(video_path)

# Load YOLO model with custom weights and push to CUDA if available
model = YOLO("Weights/best.pt")
model.to(device)

# Define class names
class_names = ['Pothole']

while True:
    success, img = cap.read()
    if not success:
        print("End of video or cannot read frame.")
        break

    # Perform detection (Ultralytics internally uses CUDA if model is on CUDA)
    results = model(img, stream=True)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            w, h = x2 - x1, y2 - y1

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            if conf > 0.4:  # Confidence threshold
                cvzone.cornerRect(img, (x1, y1, w, h), t=2)
                cvzone.putTextRect(img, f'{class_names[cls]} {conf:.2f}',
                                   (max(0, x1), max(35, y1)), scale=1, thickness=1)

    # Display output
    cv2.imshow("Pothole Detection (CUDA)" if device == 'cuda' else "Pothole Detection (CPU)", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

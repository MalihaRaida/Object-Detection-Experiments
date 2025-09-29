import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("C:\\Users\\USER\\Downloads\\archive\\yolov8n.pt")
# Load image
img = cv2.imread("C:\\Users\\USER\\Downloads\\archive\\strawberry-chocolate-cake.jpg")

# Run detection
results = model(img)

# Loop through detections
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])  # class id
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
        if cls == 55:
            roi = img[y1:y2, x1:x2]  # region of interest
            blur = cv2.GaussianBlur(roi, (51, 51), 30)
            img[y1:y2, x1:x2] = blur

# Show and save result
cv2.imshow("Blurred Image", img)
cv2.imwrite("blurred_output.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import os
from ultralytics import YOLO

model = YOLO("C:\\Users\\USER\\Downloads\\archive\\yolov8n.pt")
img = cv2.imread("C:\\Users\\USER\\Downloads\\archive\\human-img.jpg")

results = model(img)[0]
os.makedirs("crops", exist_ok=True)


for i, box in enumerate(results.boxes):
    x1, y1, x2, y2 = map(int, box.xyxy[0])   # bounding box coords
    cls = int(box.cls[0])                    # class index
    label = model.names[cls]                 # class name
    
    crop = img[y1:y2, x1:x2]                 # extract object
    save_path = f"crops/{label}_{i}.jpg"
    cv2.imwrite(save_path, crop)
    print(f"Saved {save_path}")
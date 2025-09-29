from ultralytics import YOLO


model = YOLO("yolov8n.pt")  
results = model("C:\\Users\\USER\\Downloads\\archive\\images.jpg",conf=0.5)

for result in results:
    result.show()
    boxes = result.boxes
    print(f"Detected objects in image: {result.path}")
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        confidence = box.conf[0].item()       
        class_id = int(box.cls[0].item())    
        class_name = model.names[class_id]    

        print(f"  - Object: {class_name}, Confidence: {confidence:.2f}, "
              f"Bounding Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")


# target_classes=[14,16]
# for r in results:
#     for box in r.boxes:
#         cls_id = int(box.cls.item())   # class index
#         conf   = float(box.conf.item()) # confidence
#         if cls_id in target_classes:
#             xyxy = box.xyxy[0].tolist()  # bounding box [x1, y1, x2, y2]
#             print(f"Class {cls_id}, Conf {conf:.2f}, BBox {xyxy}")


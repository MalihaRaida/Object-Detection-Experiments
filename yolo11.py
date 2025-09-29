# from ultralytics import YOLO


# model = YOLO("C:\\Users\\USER\\Downloads\\archive\\runs\\detect\\train4\\weights\\best.pt")  
# results = model("C:\\Users\\USER\\Downloads\\archive\\dog-ducks.jpg")


# for result in results:
#     result.show()
#     boxes = result.boxes
#     print(f"Detected objects in image: {result.path}")
#     for box in boxes:
#         x1, y1, x2, y2 = box.xyxy[0].tolist()
#         confidence = box.conf[0].item()       
#         class_id = int(box.cls[0].item())    
#         class_name = model.names[class_id]    

#         print(f"  - Object: {class_name}, Confidence: {confidence:.2f}, "
#               f"Bounding Box: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")


# # target_classes=[14,16]
# # for r in results:
# #     for box in r.boxes:
# #         cls_id = int(box.cls.item())   # class index
# #         conf   = float(box.conf.item()) # confidence
# #         if cls_id in target_classes:
# #             xyxy = box.xyxy[0].tolist()  # bounding box [x1, y1, x2, y2]
# #             print(f"Class {cls_id}, Conf {conf:.2f}, BBox {xyxy}")


from ultralytics import YOLO
import cv2
import numpy as np

def test_model():
    # Load your trained model
    model = YOLO('C:\\Users\\USER\\Downloads\\archive\\runs\\detect\\train5\\weights\\best.pt')
    path="C:\\Users\\USER\\Downloads\\archive\\dataset.yaml"
    # Test on validation images
    results = model.val(data=path)
    
    # Test on a specific image
    test_image = 'dog-ducks.jpg'
    results = model(test_image)

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
    
    # Visualize results
    # for r in results:
    #     # Get the original image
    #     img = r.orig_img
        
    #     # Draw bounding boxes
    #     if r.boxes is not None:
    #         boxes = r.boxes.xyxy.cpu().numpy()
    #         confidences = r.boxes.conf.cpu().numpy()
    #         class_ids = r.boxes.cls.cpu().numpy().astype(int)
            
    #         for box, conf, cls_id in zip(boxes, confidences, class_ids):
    #             x1, y1, x2, y2 = box.astype(int)
                
    #             # Get class name
    #             class_name = model.names[cls_id]
    #             label = f'{class_name}: {conf:.2f}'
                
    #             # Draw bounding box and label
    #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #             cv2.putText(img, label, (x1, y1-10), 
    #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    #     # Save or display the result
    #     cv2.imwrite('result.jpg', img)
    #     cv2.imshow('Detection Result', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    test_model()
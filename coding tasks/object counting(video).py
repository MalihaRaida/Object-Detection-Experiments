from ultralytics import YOLO
import cv2
import cvzone
import math 
from sort import *
import numpy as np


model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("C:\\Users\\USER\\Downloads\\archive\\traffic.mp4")
count = 0

mask=cv2.imread("C:\\Users\\USER\\Downloads\\archive\\mask.jpg")
start_point = (20, 450)
end_point = (915, 450)
color = (0, 255, 0)  # Green color
thickness = 5
total_counts=0
tracker=Sort(max_age=20,min_hits=2,iou_threshold=0.1)
while True:
    ret, frame = cap.read()
    if frame is None:
        break
    mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

    imageRegion=cv2.bitwise_and(frame,mask_resized)
    results=model(imageRegion,stream=True,conf=0.5)

    detections=np.empty((0,5))
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            cvzone.cornerRect(frame,(x1,y1,w,h),l=10)
            conf=math.ceil((box.conf[0]*100))/100
            cls=int(box.cls[0])
            if cls==2 or cls==7 or cls==3:
                cvzone.putTextRect(frame,f"{cls} {conf}",(max(0,x1),max(35,y1)),scale=0.5,thickness=1,offset=3)
                currentArray=np.array([x1,y1,x2,y2,conf])
                print(currentArray)
                detections=np.vstack((detections,currentArray))
    resultTracker=tracker.update(detections)

    # cv2.line(frame,start_point, end_point, color, thickness)
    print(resultTracker)
    for r in resultTracker:
        x1,y1,x2,y2,id=r
        x1,y1,x2,y2,id=int(x1),int(y1),int(x2),int(y2),int(id)
        print(r)
        w,h=x2-x1,y2-y1

        # cx,cy=x1+w//2,y1+h//2
        cvzone.putTextRect(frame,f"{id}",(max(0,x1),max(35,y1)),scale=2,thickness=1,offset=10)
        # cv2.circle(frame,(cx,cy),5,(255,0,255),cv2.FILLED)
        # if start_point[0]<cx<end_point[0] and start_point[1]-35<cy<end_point[1]+35:
        #     total_counts+=1
        # cvzone.putTextRect(frame,f"Count: {total_counts}",(50,50))
    cv2.imshow("Video",frame)
    cv2.waitKey(1)

cv2.destroyAllWindows()   
#     if not ret:
#         break

#     results = model(frame)
#     for r in results:
#         for box in r.boxes:
#             if int(box.cls[0]) == 2:
#                 count += 1

#     cv2.imshow("Video", results[0].plot())
#     if cv2.waitKey(1) == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()
# print("Total cars detected:", count)

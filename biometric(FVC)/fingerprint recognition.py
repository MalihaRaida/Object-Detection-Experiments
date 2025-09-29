import cv2
import numpy as np
import os

base_path="C:\\Users\\USER\\Downloads\\archive\\biometric(FVC)\\74034_3_En_4_MOESM1_ESM\\FVC2004\\Dbs\\DB1_A\\"
file_list= os.listdir(base_path)
print(file_list[:9])
sample=cv2.imread(base_path+file_list[0])
print(base_path+file_list[0])
fileName= None
image= None

kp1, kp2, mp= None, None, None
bestScore=0

for file in file_list[1:100]:
    print(file)
    fingerPrint_img=cv2.imread(base_path+file)
    sift=cv2.SIFT_create()

    keypoint_1,descriptor_1=sift.detectAndCompute(sample,None)
    keypoint_2,descriptor_2=sift.detectAndCompute(fingerPrint_img,None)
    matches=cv2.FlannBasedMatcher({'algorithm':1, 'trees':10},{}).knnMatch(descriptor_1,descriptor_2,k=2)
    match_points=[]
    for p, q in matches:
        if p.distance <0.3*q.distance:
            match_points.append(p)
    keypoints=0
    if len(keypoint_1)<len(keypoint_2):
        keypoints=len(keypoint_1)
    else:
        keypoints=len(keypoint_2)
    
    if len(match_points)/keypoints *100> bestScore:
        bestScore=len(match_points)/keypoints*100
        fileName= base_path+file
        image= fingerPrint_img
        kp1, kp2, mp= keypoint_1, keypoint_2, match_points

    print(bestScore)


print("Best Image" , fileName)
print(f"Score: {bestScore}")

result=cv2.drawMatches(sample,kp1,image,kp2,match_points,None)

cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()
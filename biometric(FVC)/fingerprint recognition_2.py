import cv2

import cv2


img1 = cv2.imread("C:\\Users\\USER\\Downloads\\archive\\biometric(FVC)\\74034_3_En_4_MOESM1_ESM\\FVC2004\\Dbs\\DB1_A\\100_1.tif", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("C:\\Users\\USER\\Downloads\\archive\\biometric(FVC)\\74034_3_En_4_MOESM1_ESM\\FVC2004\\Dbs\\DB1_A\\3_2.tif", cv2.IMREAD_GRAYSCALE)
match_threshold=20
orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)


matches = sorted(matches, key=lambda x: x.distance)
good_matches = [m for m in matches if m.distance < 50]
print(f"Total Matches: {len(matches)} | Good Matches: {len(good_matches)}")

is_match = len(good_matches) > match_threshold

matched_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:20], None, flags=2)
cv2.imshow("Matches", matched_img)
print("Match?" , is_match)
cv2.waitKey(0)
cv2.destroyAllWindows()


# matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
# cv2.imshow("Matches", matched_img)
# cv2.waitKey(0)


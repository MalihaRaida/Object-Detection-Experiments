# import cv2
# import numpy as np
# from skimage.filters import threshold_otsu
# from skimage.transform import resize
# import matplotlib.pyplot as plt

# # Load the eye image
# img = cv2.imread('C:\\Users\\USER\\Downloads\\Retica_sample_iris_DB\\0001_L_000.png', cv2.IMREAD_GRAYSCALE)

# # Step 1: Pupil segmentation (using threshold)
# _, pupil_mask = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
# pupil_mask = cv2.morphologyEx(pupil_mask, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))

# # Find pupil contour
# contours, _ = cv2.findContours(pupil_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# pupil_contour = max(contours, key=cv2.contourArea)
# (x, y), pupil_radius = cv2.minEnclosingCircle(pupil_contour)
# pupil_center = (int(x), int(y))

# # Step 2: Iris segmentation (roughly by intensity threshold)
# thresh = threshold_otsu(img)
# iris_mask = img < thresh

# # Step 3: Combine masks to isolate iris
# iris_only = cv2.bitwise_and(img, img, mask=iris_mask.astype(np.uint8)*255)

# # Step 4: Normalization (unwrap iris into rectangular strip)
# def normalize_iris(img, center, rp, ri, radpixels=64, angres=512):
#     theta = np.linspace(0, 2*np.pi, angres)
#     r = np.linspace(rp, ri, radpixels)
#     polar_array = np.zeros((radpixels, angres))
#     for i in range(angres):
#         for j in range(radpixels):
#             x = int(center[0] + r[j]*np.cos(theta[i]))
#             y = int(center[1] + r[j]*np.sin(theta[i]))
#             if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
#                 polar_array[j, i] = img[y, x]
#     return polar_array

# normalized = normalize_iris(img, pupil_center, pupil_radius, pupil_radius*2)

# # Step 5: Feature extraction (simple Gabor filter)
# import cv2
# kernel = cv2.getGaborKernel((21, 21), 3.0, 1*np.pi/4, 10.0, 0.5, 0, ktype=cv2.CV_32F)
# features = cv2.filter2D(normalized, cv2.CV_8UC3, kernel)

# # Step 6: Display results
# plt.figure(figsize=(10,5))
# plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Eye')
# plt.subplot(132), plt.imshow(iris_only, cmap='gray'), plt.title('Segmented Iris')
# plt.subplot(133), plt.imshow(features, cmap='gray'), plt.title('Extracted Features')
# plt.show()

# import cv2
# import numpy as np
# from skimage.filters import gabor
# from skimage import io, color, img_as_float
# import matplotlib.pyplot as plt

# # Step 1: Load the image
# img = cv2.imread("C:\\Users\\USER\\Downloads\\Human_eye_iris_5.jpg",cv2.IMREAD_GRAYSCALE)
# gray = img

# # Step 2: Detect the iris using Hough Circles (simplified)
# blur = cv2.medianBlur((gray * 255).astype(np.uint8), 5)
# circles = cv2.HoughCircles(
#     blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
#     param1=80, param2=30, minRadius=30, maxRadius=80
# )



# if circles is not None:
#     circles = np.uint16(np.around(circles))
#     pupil_x, pupil_y, pupil_r = circles[0, 0]

# iris_circles = cv2.HoughCircles(
#     blur, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
#     param1=100, param2=30, minRadius=pupil_r + 50, maxRadius=pupil_r + 100
# )

# if iris_circles is not None:
#     iris_circles = np.uint16(np.around(iris_circles))
#     iris_x, iris_y, iris_r = iris_circles[0, 0]

# mask = np.zeros_like(gray)
# cv2.circle(mask, (iris_x, iris_y), iris_r, 255, -1)
# cv2.circle(mask, (pupil_x, pupil_y), pupil_r, 0, -1)
# iris_region = cv2.bitwise_and(gray, gray, mask=mask)

# # Step 5: Crop region of interest
# y1, y2 = iris_y - iris_r, iris_y + iris_r
# x1, x2 = iris_x - iris_r, iris_x + iris_r
# iris_crop = iris_region[y1:y2, x1:x2]

# # Step 6: Show results
# output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# cv2.circle(output, (pupil_x, pupil_y), pupil_r, (0, 255, 0), 2)
# cv2.circle(output, (iris_x, iris_y), iris_r, (255, 0, 0), 2)

# plt.figure(figsize=(10, 5))
# plt.subplot(1, 2, 1)
# plt.title("Detected Circles")
# plt.imshow(output)
# plt.axis("off")

# plt.subplot(1, 2, 2)
# plt.title("Extracted Iris Region")
# plt.imshow(iris_crop, cmap='gray')
# plt.axis("off")

# plt.show()



# Step 3: Normalize iris (resize)
# normalized_iris = cv2.resize(iris, (128, 64))

# # Step 4: Feature extraction (Gabor filter)
# filt_real, filt_imag = gabor(normalized_iris, frequency=0.6)
# iris_features = np.abs(filt_real + 1j * filt_imag)

# # Step 5: Display results
# plt.subplot(1, 3, 1); plt.title("Original"); plt.imshow(gray, cmap='gray')
# plt.subplot(1, 3, 2); plt.title("Iris Region"); plt.imshow(iris, cmap='gray')
# plt.subplot(1, 3, 3); plt.title("Features"); plt.imshow(iris_features, cmap='gray')
# plt.show()


import cv2
import numpy as np
from skimage.filters import gabor
import matplotlib.pyplot as plt

#  Read image and preprocess
img = cv2.imread('C:\\Users\\USER\\Downloads\\Human_eye_iris_5.jpg', cv2.IMREAD_GRAYSCALE)
img_blur = cv2.medianBlur(img, 5)

#  Detect the pupil (darkest circular region)
_, thresh = cv2.threshold(img_blur, 40, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest dark circle-like region (pupil)
max_contour = max(contours, key=cv2.contourArea)
(x_pupil, y_pupil), pupil_radius = cv2.minEnclosingCircle(max_contour)
center_pupil = (int(x_pupil), int(y_pupil))
pupil_radius = int(pupil_radius)

#  Detect iris outer boundary via edge detection
edges = cv2.Canny(img_blur, 40, 100)
# Use circular region growing or find largest contour around pupil
iris_contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
iris_contour = max(iris_contours, key=cv2.contourArea)
(x_iris, y_iris), iris_radius = cv2.minEnclosingCircle(iris_contour)
center_iris = (int(x_iris), int(y_iris))
iris_radius = int(iris_radius)

img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.circle(img_vis, center_pupil, pupil_radius, (0, 255, 0), 2)
cv2.circle(img_vis, center_iris, iris_radius, (255, 0, 0), 2)


# Normalize (unwrap iris using polar coordinates)
def normalize_iris(img, center, r_in, r_out, h=64, w=512):
    theta = np.linspace(0, 2*np.pi, w)
    r = np.linspace(r_in, r_out, h)
    output = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            x = center[0] + r[i] * np.cos(theta[j])
            y = center[1] + r[i] * np.sin(theta[j])
            if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
                output[i, j] = img[int(y), int(x)]
    return output

normalized = normalize_iris(img, center_pupil, pupil_radius, iris_radius)

# def extract_features(normalized_img):
#     f_real, f_imag = gabor(normalized_img, frequency=0.6)
#     return (f_real > 0).astype(np.uint8)  # Simple binary iris code

# iris_code = extract_features(normalized)

# # Matching (Hamming distance between two iris codes)
# def match_iris(code1, code2):
#     return np.sum(code1 == code2) / code1.size

# Example: comparing two iris images
# score = match_iris(iris_code1, iris_code2)

f_real, f_imag = gabor(normalized, frequency=0.6)
features = (f_real > 0).astype(np.uint8)

# Visualization
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title("Original Eye")
plt.imshow(img, cmap='gray')

plt.subplot(2, 3, 2)
plt.title("Pupil Threshold")
plt.imshow(thresh, cmap='gray')

plt.subplot(2, 3, 3)
plt.title("Edges")
plt.imshow(edges, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("Detected Pupil + Iris")
plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))

plt.subplot(2, 3, 5)
plt.title("Normalized Iris (Unwrapped)")
plt.imshow(normalized, cmap='gray')

plt.subplot(2, 3, 6)
plt.title("Iris Features (Gabor)")
plt.imshow(features, cmap='gray')

plt.tight_layout()
plt.show()

import cv2
from ultralytics import YOLO
import numpy as np


def gaussian_kernel(size, sigma=1):
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def convolve2d(image, kernel):
    pad = kernel.shape[0] // 2
    padded = np.pad(image, pad, mode='reflect')
    output = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[i:i+kernel.shape[0], j:j+kernel.shape[1]]
            output[i, j] = np.sum(region * kernel)
    return output


# Load model (pretrained COCO)
model = YOLO("C:\\Users\\USER\\Downloads\\archive\\yolov8n.pt")
# Load image
img = cv2.imread("C:\\Users\\USER\\Downloads\\archive\\strawberry-chocolate-cake.jpg")

# Run detection
results = model(img)[0]


# blurred = cv2.GaussianBlur(img, (51, 51), 0)
kernel = gaussian_kernel(size=51, sigma=10)
blurred = np.zeros_like(img)
for c in range(img.shape[2]):
    blurred[:, :, c] = convolve2d(img[:, :, c], kernel)




# Keep detected objects clear
for box in results.boxes:
    cls = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # bounding box
    if cls == 55:
        roi = img[y1:y2, x1:x2]                 # object region from original
        blurred[y1:y2, x1:x2] = roi             # paste onto blurred image

# Show result

cv2.namedWindow('Resizable Image', cv2.WINDOW_NORMAL)
cv2.imshow('Resizable Image', blurred)

cv2.imwrite("blurred_output_3.jpg", blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()

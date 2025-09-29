import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from torchvision.transforms import functional as F
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from PIL import Image


# model = fasterrcnn_resnet50_fpn(pretrained=True)
# num_classes = 3

# in_features = model.roi_heads.box_predictor.cls_score.in_features
# model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# transform = T.Compose([
#     T.ToTensor(),
# ])

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# img = Image.open("C:\\Users\\USER\\Downloads\\archive\\dog-ducks-6-1-3dd027ec854047d7b734df7a21cae2f5.jpg")
# img = transform(img)
# img = img.unsqueeze_(0).to(device)
# # Model prediction
# model.eval()
# with torch.no_grad():
#     prediction = model([img])
# # Print the predicted bounding boxes and labels
# print(prediction[0]['boxes'])
# print(prediction[0]['labels'])

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

image_path = "C:\\Users\\USER\\Downloads\\archive\\dog-ducks-6-1-3dd027ec854047d7b734df7a21cae2f5.jpg" 
image= Image.open(image_path).convert("RGB")

img_tensor = F.to_tensor(image).unsqueeze(0) 

with torch.no_grad():
    predictions = model(img_tensor)

boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']


threshold = 0.5
keep = scores >= threshold

boxes = boxes[keep]
labels = labels[keep]
scores = scores[keep]


fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

for box, label, score in zip(boxes, labels, scores):
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin, f"{label.item()}:{score:.2f}", 
            fontsize=10, color='white', 
            bbox=dict(facecolor='red', alpha=0.5))

plt.show()

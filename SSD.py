import torch
import torchvision
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


model = torchvision.models.detection.ssd300_vgg16(pretrained=True)
model.eval()


img_path = "C:\\Users\\USER\\Downloads\\human-emotions-and-feeligs-concept-photo.jpg"  # replace with your image path
image = Image.open(img_path).convert("RGB")

img_tensor = F.to_tensor(image).unsqueeze(0)


with torch.no_grad():
    outputs = model(img_tensor)


boxes = outputs[0]["boxes"]
labels = outputs[0]["labels"]
scores = outputs[0]["scores"]


threshold = 0.5
keep = scores >= threshold
boxes, labels, scores = boxes[keep], labels[keep], scores[keep]

COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

fig, ax = plt.subplots(1, figsize=(12, 9))
ax.imshow(image)

for box, label, score in zip(boxes, labels, scores):
    xmin, ymin, xmax, ymax = box
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                             linewidth=2, edgecolor="red", facecolor="none")
    ax.add_patch(rect)
    ax.text(xmin, ymin, f"{COCO_CLASSES[label]}:{score:.2f}",
            fontsize=10, color="white", bbox=dict(facecolor="red", alpha=0.5))
    print(f"Label {label}")

plt.show()

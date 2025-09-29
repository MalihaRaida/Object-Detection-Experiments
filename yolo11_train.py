# from ultralytics import YOLO


# model = YOLO("yolo11n.pt")

# path="C:\\Users\\USER\\Downloads\\archive\\dataset.yaml"

# train_results = model.train(
#     data=path,  # Path to dataset configuration file
#     epochs=50  # Number of training epochs
#     froz
# )


from ultralytics import YOLO
import torch

path="C:\\Users\\USER\\Downloads\\archive\\dataset.yaml" 

def train_yolo_with_duck():
    # Load pre-trained YOLO11 model
    model = YOLO('yolo11n.pt')  # or yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
    
    # Check if GPU is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    # Training parameters
    results = model.train(
        data=path,  # Path to your dataset config
        epochs=50,                # Number of training epochs
        imgsz=640,                # Image size
        batch=16,                 # Batch size (adjust based on your GPU memory)
        lr0=0.01,                 # Initial learning rate
        device=device,
        patience=10,              # Early stopping patience
        save_period=10,           # Save model every 10 epochs
        pretrained=True,          # Use pretrained weights
        freeze=None,              # Don't freeze any layers (fine-tune all)
        
        # Data augmentation
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )
    
    # Return the trained model
    return model

if __name__ == "__main__":
    trained_model = train_yolo_with_duck()
    print("Training completed!")
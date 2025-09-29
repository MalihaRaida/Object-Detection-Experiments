import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import os
import time
from pathlib import Path

class SmallVideoObjectCounter:
    def __init__(self, model_size='n', confidence_threshold=0.5):
        """
        Initialize the small video object counter
        
        Args:
            model_size (str): Model size - 'n'(nano), 's'(small), 'm'(medium)
            confidence_threshold (float): Confidence threshold for detections
        """
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.confidence_threshold = confidence_threshold
        self.class_names = self.model.names
        
        # Generate colors for each class
        np.random.seed(42)
        self.colors = {}
        for class_id in self.class_names.keys():
            self.colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
        
        # Counting variables
        self.frame_counts = []
        self.max_counts = defaultdict(int)
        
    def process_small_video(self, video_path, output_path=None, show_video=True):
        """
        Process a small MP4 video and count objects
        
        Args:
            video_path (str): Path to the input MP4 video
            output_path (str): Path to save the output video (optional)
            show_video (bool): Whether to display the video while processing
        
        Returns:
            dict: Final object counts and statistics
        """
        # Validate input file
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        if not video_path.lower().endswith('.mp4'):
            print("Warning: File doesn't have .mp4 extension")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"📹 Processing: {Path(video_path).name}")
        print(f"📊 Video info: {width}x{height}, {fps}FPS, {duration:.1f}s, {total_frames} frames")
        
        # Setup video writer
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"💾 Output will be saved to: {output_path}")
        
        # Processing variables
        frame_num = 0
        start_time = time.time()
        
        print("\n🔄 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            
            # Run YOLO detection
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
            
            # Count objects in current frame
            current_counts = defaultdict(int)
            
            # Process detections
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get detection info
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.class_names[class_id]
                        
                        # Count this detection
                        current_counts[class_name] += 1
                        
                        # Draw bounding box
                        color = self.colors[class_id]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        
                        # Draw label with confidence
                        label = f"{class_name} {confidence:.2f}"
                        (label_width, label_height), _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        
                        cv2.rectangle(frame, (x1, y1 - label_height - 10), 
                                    (x1 + label_width, y1), color, -1)
                        cv2.putText(frame, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Update maximum counts
            for class_name, count in current_counts.items():
                self.max_counts[class_name] = max(self.max_counts[class_name], count)
            
            # Store frame counts
            self.frame_counts.append(dict(current_counts))
            
            # Add statistics overlay
            self._draw_stats_overlay(frame, current_counts, frame_num, total_frames)
            
            # Show video
            if show_video:
                cv2.imshow('Object Counter - Press Q to quit', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("⏹️ Stopped by user")
                    break
            
            # Save frame
            if out:
                out.write(frame)
            
            # Progress update every 30 frames
            if frame_num % 30 == 0 or frame_num == total_frames:
                progress = (frame_num / total_frames) * 100
                elapsed = time.time() - start_time
                fps_current = frame_num / elapsed
                print(f"Progress: {progress:.1f}% | Frame: {frame_num}/{total_frames} | FPS: {fps_current:.1f}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if show_video:
            cv2.destroyAllWindows()
        
        # Calculate and display final results
        processing_time = time.time() - start_time
        results = self._generate_final_report(video_path, frame_num, processing_time)
        
        return results
    
    def _draw_stats_overlay(self, frame, current_counts, frame_num, total_frames):
        """Draw statistics overlay on frame"""
        # Create semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "Object Counter", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Frame info
        cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Current detections
        y_pos = 75
        if current_counts:
            for class_name, count in current_counts.items():
                text = f"{class_name}: {count}"
                cv2.putText(frame, text, (20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_pos += 18
        else:
            cv2.putText(frame, "No objects detected", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def _generate_final_report(self, video_path, frames_processed, processing_time):
        """Generate and display final counting report"""
        print("\n" + "="*60)
        print("🎯 FINAL OBJECT COUNTING RESULTS")
        print("="*60)
        
        print(f"📁 Video: {Path(video_path).name}")
        print(f"⏱️ Processing time: {processing_time:.2f} seconds")
        print(f"📊 Frames processed: {frames_processed}")
        print(f"⚡ Average FPS: {frames_processed/processing_time:.1f}")
        
        print("\n📈 Maximum object counts detected:")
        print("-" * 40)
        
        if self.max_counts:
            for class_name in sorted(self.max_counts.keys()):
                count = self.max_counts[class_name]
                print(f"  {class_name:15s}: {count:3d}")
        else:
            print("  No objects detected in the video")
        
        # Calculate some statistics
        total_detections = sum(len(frame_count) for frame_count in self.frame_counts)
        frames_with_objects = sum(1 for frame_count in self.frame_counts if frame_count)
        
        print(f"\n📊 Additional statistics:")
        print(f"  Total detections: {total_detections}")
        print(f"  Frames with objects: {frames_with_objects}/{frames_processed}")
        print(f"  Detection rate: {(frames_with_objects/frames_processed)*100:.1f}%")
        
        # Save report to file
        report_path = Path(video_path).stem + "_count_report.txt"
        self._save_report(report_path, video_path, frames_processed, processing_time)
        print(f"\n💾 Detailed report saved to: {report_path}")
        
        return {
            'max_counts': dict(self.max_counts),
            'total_detections': total_detections,
            'frames_processed': frames_processed,
            'frames_with_objects': frames_with_objects,
            'processing_time': processing_time
        }
    
    def _save_report(self, report_path, video_path, frames_processed, processing_time):
        """Save detailed report to text file"""
        with open(report_path, 'w') as f:
            f.write("YOLOv8 Small Video Object Counting Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Video file: {video_path}\n")
            f.write(f"Processing date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: YOLOv8 (confidence: {self.confidence_threshold})\n")
            f.write(f"Processing time: {processing_time:.2f} seconds\n")
            f.write(f"Frames processed: {frames_processed}\n\n")
            
            f.write("Maximum Object Counts:\n")
            f.write("-" * 25 + "\n")
            for class_name, count in sorted(self.max_counts.items()):
                f.write(f"{class_name}: {count}\n")
            
            f.write(f"\nFrame-by-frame counts (last 20 frames):\n")
            f.write("-" * 35 + "\n")
            for i, frame_count in enumerate(self.frame_counts[-20:], 1):
                frame_num = len(self.frame_counts) - 20 + i
                if frame_count:
                    f.write(f"Frame {frame_num}: {frame_count}\n")


def main():
    """Simple main function for easy video processing"""
    print("🚀 YOLOv8 Small MP4 Video Object Counter")
    print("=" * 45)
    
    # Get video path from user
    video_path = "C:\\Users\\USER\\Downloads\\archive\\traffic.mp4"
    
    
    # Ask for output video (optional)
    output_path = "C:\\Users\\USER\\Downloads\\archive\\traffic_counted.mp4"
    
    # Ask for model size
    
    model_size = 'n'
    
    # Ask for confidence threshold
    
    confidence = 0.5
    
    try:
        # Initialize counter
        print(f"\n⚙️ Initializing YOLOv8{model_size} model...")
        counter = SmallVideoObjectCounter(model_size=model_size, confidence_threshold=confidence)
        
        # Process video
        print("🔄 Starting video processing...")
        results = counter.process_small_video(
            video_path=video_path,
            output_path=output_path,
            show_video=True
        )
        
        print("\n✅ Processing completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ File error: {e}")
    except Exception as e:
        print(f"❌ Error during processing: {e}")
    
    print("\n👋 Press any key to exit...")
    input()


if __name__ == "__main__":
    main()
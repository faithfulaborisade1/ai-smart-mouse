import torch
import numpy as np
import cv2
import time

class YOLODetector:
    def __init__(self, model_path='yolov5s.pt', conf_threshold=0.5, device='cuda:0'):
        """
        Initialize the YOLO detector
        
        Args:
            model_path: Path to the YOLOv5 model file
            conf_threshold: Confidence threshold for detections
            device: Device to run inference on ('cuda:0' or 'cpu')
        """
        # Load YOLOv5 model
        try:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        except Exception as e:
            print(f"Failed to load custom model: {e}")
            print("Loading default YOLOv5s model instead...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        
        # Set model parameters
        self.model.conf = conf_threshold  # Confidence threshold
        
        # Set device (GPU or CPU)
        try:
            self.model.to(torch.device(device))
            print(f"Using device: {device}")
        except:
            print(f"Device {device} not available, using CPU instead")
            self.model.to(torch.device('cpu'))
        
        # Get class names
        self.class_names = self.model.names
        print(f"Loaded model with classes: {self.class_names}")
        
        # For FPS calculation
        self.prev_time = time.time()
        self.fps = 0
    
    def detect(self, frame):
        """
        Detect objects in the given frame
        
        Args:
            frame: Input image frame (numpy array)
            
        Returns:
            boxes: List of detection boxes [x, y, w, h]
            confidences: Confidence scores for each box
            class_ids: Class IDs for each box
        """
        # Measure FPS
        current_time = time.time()
        self.fps = 1 / (current_time - self.prev_time)
        self.prev_time = current_time
        
        # Convert to RGB (YOLOv5 expects RGB, OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference
        results = self.model(rgb_frame)
        
        # Process results
        predictions = results.xyxy[0].cpu().numpy()  # Get predictions as numpy array
        
        boxes = []
        confidences = []
        class_ids = []
        
        for pred in predictions:
            x1, y1, x2, y2, conf, class_id = pred
            
            # Convert to [x, y, w, h] format
            x, y = int(x1), int(y1)
            w, h = int(x2 - x1), int(y2 - y1)
            
            boxes.append([x, y, w, h])
            confidences.append(float(conf))
            class_ids.append(int(class_id))
        
        return boxes, confidences, class_ids
    
    def draw_detections(self, frame, boxes, confidences, class_ids):
        """
        Draw detection boxes on the frame
        
        Args:
            frame: Input image frame
            boxes: Detection boxes [x, y, w, h]
            confidences: Confidence scores
            class_ids: Class IDs
            
        Returns:
            Annotated frame
        """
        result_frame = frame.copy()
        
        for i, box in enumerate(boxes):
            x, y, w, h = box
            conf = confidences[i]
            class_id = class_ids[i]
            
            # Get class name
            class_name = self.class_names[class_id]
            
            # Select color based on class_id (cyclic)
            color = (0, 255, 0) if class_name == 'person' else (0, 165, 255)
            
            # Draw box
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            cv2.putText(result_frame, label, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add FPS
        cv2.putText(result_frame, f"FPS: {self.fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return result_frame
    
    def filter_player_detections(self, boxes, confidences, class_ids):
        """
        Filter detections to only include players
        
        Args:
            boxes: Detection boxes
            confidences: Confidence scores
            class_ids: Class IDs
            
        Returns:
            Filtered boxes, confidences, class_ids
        """
        player_indices = [i for i, class_id in enumerate(class_ids) 
                         if self.class_names[class_id] == 'person']
        
        filtered_boxes = [boxes[i] for i in player_indices]
        filtered_confidences = [confidences[i] for i in player_indices]
        filtered_class_ids = [class_ids[i] for i in player_indices]
        
        return filtered_boxes, filtered_confidences, filtered_class_ids

# Training guidance for custom YOLOv5 model
"""
To train a custom YOLOv5 model for your game:

1. Collect screenshots (~500-1000) from your game with enemies visible
2. Annotate them using a tool like labelImg (https://github.com/tzutalin/labelImg)
   - Save annotations in YOLO format
3. Organize your data following YOLOv5 requirements:
   /dataset
     /images
       /train
       /val
     /labels
       /train
       /val
4. Create a dataset.yaml file:
   train: ./dataset/images/train
   val: ./dataset/images/val
   nc: 1  # Number of classes
   names: ['enemy']  # Class names
5. Train the model:
   !python train.py --img 640 --batch 16 --epochs 100 --data dataset.yaml --weights yolov5s.pt
6. Use the resulting best.pt model in this detector

For games like Valorant or Apex, you may need to fine-tune on specific character models.
"""

# Example usage
if __name__ == "__main__":
    # Load test image
    test_image = cv2.imread("test_game_screenshot.jpg")
    
    # Create detector and detect objects
    detector = YOLODetector(device='cpu')  # Use GPU if available
    boxes, confidences, class_ids = detector.detect(test_image)
    
    # Filter for player detections
    player_boxes, player_confs, player_ids = detector.filter_player_detections(
        boxes, confidences, class_ids
    )
    
    # Draw detections
    result = detector.draw_detections(test_image, player_boxes, player_confs, player_ids)
    
    # Display result
    cv2.imshow("YOLO Detections", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
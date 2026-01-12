from ultralytics import YOLO
import cv2
import numpy as np

class LicensePlateDetector:
    def __init__(self, model_path='license_plate_detector.pt'):
        # Fallback to yolov8n.pt if default custom model isn't found, but prefer custom.
        if model_path == 'license_plate_detector.pt':
             import os
             if not os.path.exists(model_path):
                 print("Custom model not found locally. Using generic yolov8n.pt (will detect cars, not plates).")
                 model_path = 'yolov8n.pt'
        
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        
        # Analyze model classes
        self.names = self.model.names
        print(f"Model classes: {self.names}")
        
        # Determine which class ID corresponds to license plates
        self.target_class_ids = []
        
        # Heuristics to find the plate class
        kw = ['plate', 'license']
        for cls_id, cls_name in self.names.items():
            if any(k in cls_name.lower() for k in kw):
                self.target_class_ids.append(cls_id)
        
        # If no specific name found but it's a custom 1-class model, assume it's the plate
        if not self.target_class_ids and len(self.names) == 1:
            self.target_class_ids = [0]
            print("Single class model detected. Assuming class 0 is license plate.")
            
        print(f"Targeting class IDs: {self.target_class_ids}")

    def detect(self, frame):
        """
        Detects license plates in the frame.
        Returns a list of bounding boxes [x1, y1, x2, y2].
        """
        # Run inference
        results = self.model(frame, verbose=False)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                
                # If we have target classes, filter by them. 
                # If we have NO target classes (standard model), detection does nothing (or we could debug-detect cars).
                if cls_id in self.target_class_ids:
                     x1, y1, x2, y2 = map(int, box.xyxy[0])
                     boxes.append((x1, y1, x2, y2))
                
        return boxes

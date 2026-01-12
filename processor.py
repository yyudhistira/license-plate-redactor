import cv2
import time
from detector import LicensePlateDetector

class VideoProcessor:
    def __init__(self, detector: LicensePlateDetector):
        self.detector = detector

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Processing VIDEO: {input_path}")
        print(f"Dimensions: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

        # Setup codec and writer
        # mp4v is broadly compatible. avc1 is often preferred but might require openh264 on some systems.
        # We'll try mp4v first as it's safe.
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        start_time = time.time()

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect license plates
                boxes = self.detector.detect(frame)

                # Redact (blur) the plates
                for box in boxes:
                    x1, y1, x2, y2 = box
                    # Ensure coordinates are within image bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    
                    # Extract ROI
                    roi = frame[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Apply heavy gaussian blur or solid grey
                        # Option 1: Solid Grey Box (as requested "grey box")
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), -1)
                        
                        # Option 2: Pixelate or Blur (commented out)
                        # roi = cv2.GaussianBlur(roi, (51, 51), 30)
                        # frame[y1:y2, x1:x2] = roi

                out.write(frame)
                
                frame_count += 1
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_proc = frame_count / elapsed
                    percent = (frame_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Progress: {percent:.1f}% ({frame_count}/{total_frames}) - Speed: {fps_proc:.1f} FPS", end='\r')

        finally:
            cap.release()
            out.release()
            print("\nProcessing complete.")

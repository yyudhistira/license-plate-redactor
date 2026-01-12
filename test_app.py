import cv2
import numpy as np
import os
from processor import VideoProcessor
from detector import LicensePlateDetector

def create_dummy_video(filename, width=640, height=480, fps=30, duration=2):
    print(f"Creating dummy video: {filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    frames = int(duration * fps)
    for i in range(frames):
        # Create a frame with random noise
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        
        # Draw a fake "plate" that won't be detected but ensures visual variance
        cv2.rectangle(frame, (100+i, 100), (300+i, 150), (255, 255, 255), -1)
        
        out.write(frame)
    out.release()

def test_pipeline():
    dummy_input = "test_input.mp4"
    dummy_output = "test_output.mp4"
    
    try:
        # 1. Create dummy input
        create_dummy_video(dummy_input)
        
        # 2. Init detector
        print("Initializing Detector...")
        detector = LicensePlateDetector()
        
        # 3. Init processor
        print("Initializing Processor...")
        processor = VideoProcessor(detector)
        
        # 4. Run processing
        print("Running Processing...")
        processor.process_video(dummy_input, dummy_output)
        
        # 5. Verify output exists
        if os.path.exists(dummy_output) and os.path.getsize(dummy_output) > 0:
            print("SUCCESS: Output video created.")
        else:
            print("FAILURE: Output video missing or empty.")
            
    except Exception as e:
        print(f"FAILURE with error: {e}")
    finally:
        # Cleanup
        if os.path.exists(dummy_input):
            os.remove(dummy_input)
        if os.path.exists(dummy_output):
            os.remove(dummy_output)

if __name__ == "__main__":
    test_pipeline()

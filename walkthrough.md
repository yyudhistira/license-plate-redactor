# License Plate Redactor Walkthrough

I have successfully created the License Plate Redactor application. This CLI tool uses YOLOv8 to detect license plates in videos and redact them with a grey box.

## Setup

1.  **Dependencies**: The required libraries are listed in [requirements.txt](file:///Users/yasri/.gemini/antigravity/scratch/license-plate-redactor/requirements.txt).
    ```bash
    pip install -r requirements.txt
    ```
2.  **Model**: A pre-trained license plate detection model (`license_plate_detector.pt`) has been downloaded to the project directory.

## Usage

To redact license plates in a video file:

```bash
python main.py input_video.mp4 output_video.mp4
```

### Options
- `--model`: Path to a custom YOLO model (defaults to `license_plate_detector.pt`).

## Verification Results

I verified the application using a generated test script [test_app.py](file:///Users/yasri/.gemini/antigravity/scratch/license-plate-redactor/test_app.py), which:
1.  Created a synthetic video.
2.  Initialized the [LicensePlateDetector](file:///Users/yasri/.gemini/antigravity/scratch/license-plate-redactor/detector.py#5-57) with the downloaded model.
3.  Processed the video through [VideoProcessor](file:///Users/yasri/.gemini/antigravity/scratch/license-plate-redactor/processor.py#5-73).
4.  Confirmed the output video was created successfully.

**Test Output:**
```text
Creating dummy video: test_input.mp4
Initializing Detector...
Loading YOLO model from license_plate_detector.pt...
Model classes: {0: 'license_plate'}
Targeting class IDs: [0]
Initializing Processor...
Running Processing...
Processing VIDEO: test_input.mp4
Dimensions: 640x480, FPS: 30.0, Total Frames: 60
Progress: 100.0% (60/60) - Speed: 24.9 FPS
Processing complete.
SUCCESS: Output video created.
```

## Next Steps

-   **Performance Optimization**: If processing is too slow on very large files, consider resizing frames before detection (inference on smaller size, scale boxes back up).
-   **Model Improvement**: If the current model misses plates, you can train a custom YOLOv8 model on a larger and more specific dataset.

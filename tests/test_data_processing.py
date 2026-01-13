import os
import pytest
from detector import LicensePlateDetector
from processor import VideoProcessor

def get_test_videos():
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    if not os.path.exists(data_dir):
        return []
    return [f for f in os.listdir(data_dir) if f.lower().endswith(".mp4")]

@pytest.mark.parametrize("video_file", get_test_videos())
def test_process_data_video(video_file):
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    input_path = os.path.join(base_dir, "tests", "data", video_file)
    output_dir = os.path.join(base_dir, "tests", "output")
    output_path = os.path.join(output_dir, f"redacted_{video_file}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    detector = LicensePlateDetector(model_path=os.path.join(base_dir, "license_plate_detector.pt"))
    processor = VideoProcessor(detector)
    
    # Process video
    processor.process_video(input_path, output_path)
    
    # Verification
    assert os.path.exists(output_path), f"Output file {output_path} was not created"
    assert os.path.getsize(output_path) > 0, f"Output file {output_path} is empty"

if __name__ == "__main__":
    # If run directly, process all videos
    videos = get_test_videos()
    if not videos:
        print("No videos found in tests/data")
    else:
        for video in videos:
            print(f"Testing {video}...")
            test_process_data_video(video)
            print(f"Finished {video}")

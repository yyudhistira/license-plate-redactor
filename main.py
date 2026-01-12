import argparse
import sys
import os
from detector import LicensePlateDetector
from processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser(description="License Plate Redactor")
    parser.add_argument("input_file", help="Path to input MP4 video file")
    parser.add_argument("output_file", help="Path to output MP4 video file")
    parser.add_argument("--model", default="license_plate_detector.pt", help="Path to YOLO model file (default: license_plate_detector.pt)")

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    print("Initializing detector...")
    # In a real scenario, we would use a model fine-tuned for license plates.
    # For now, we will use the generic model. 
    # NOTE: Standard YOLOv8n includes 'car', 'truck', 'bus', but NOT 'license plate'.
    # To make this functional for the demo without a custom model, we might need to 
    # rely on a specific 'license plate' class if we find a model, or we can just 
    # demonstrate the pipeline by blurring 'cars' temporarily if no plate model is present.
    # ideally, user should provide a path to a license plate model.
    detector = LicensePlateDetector(model_path=args.model)

    processor = VideoProcessor(detector)
    
    try:
        processor.process_video(args.input_file, args.output_file)
        print(f"Success! Output saved to {args.output_file}")
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

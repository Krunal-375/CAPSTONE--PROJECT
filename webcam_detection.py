from ultralytics import YOLO
import cv2
import numpy as np

def main():
    # Load the YOLOv11 model
    model = YOLO('best.pt')  # load a pre-trained model

    # Try different camera indices
    for camera_index in [0, 1]:
        print(f"Trying camera index {camera_index}...")
        cap = cv2.VideoCapture(camera_index)
        
        if cap.isOpened():
            print(f"Successfully opened camera {camera_index}")
            break
        else:
            print(f"Failed to open camera {camera_index}")
            cap.release()
    
    if not cap.isOpened():
        print("\nError: Could not open any webcam. Please check:")
        print("1. System Preferences -> Security & Privacy -> Camera")
        print("2. Make sure your webcam isn't being used by another application")
        print("3. Try disconnecting and reconnecting your webcam if it's external")
        return

    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break

        # Run YOLOv11 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow('Sign Language Detection', annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the webcam and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

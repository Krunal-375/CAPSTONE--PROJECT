from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np
from threading import Thread, Lock
import time

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = Lock()

@app.route('/')
def index():
    """Route for the home page"""
    return render_template('index.html')

def detect_signs():
    """Function to run sign language detection on video frames"""
    global camera, output_frame, lock
    
    # Load the YOLOv8 model
    model = YOLO('best.pt')
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Ensure thread-safe access to the output frame
        with lock:
            output_frame = annotated_frame.copy()

def generate():
    """Video streaming generator function"""
    global output_frame, lock
    
    while True:
        with lock:
            if output_frame is None:
                continue
            
            # Encode the frame in JPEG format
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
            
        # Yield the output frame in byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        time.sleep(0.033)  # ~30 FPS

@app.route('/video_feed')
def video_feed():
    """Route for video streaming"""
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def initialize_camera():
    """Initialize the camera and detection thread"""
    global camera
    
    # Initialize the video capture
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera")
        return False
    
    # Check if we can read from camera
    ret, frame = camera.read()
    if not ret:
        print("Error: Could not read from camera")
        camera.release()
        return False
        
    print("Camera initialized successfully")
    print(f"Frame size: {frame.shape}")
    
    # Start the detection thread
    detection_thread = Thread(target=detect_signs, args=())
    detection_thread.daemon = True
    detection_thread.start()
    return True

if __name__ == '__main__':
    # Initialize the camera and detection thread
    if initialize_camera():
        print("Starting Flask server...")
        # Run the Flask app
        app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
    else:
        print("Failed to initialize camera. Please check your webcam connection.")

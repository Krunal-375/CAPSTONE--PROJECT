from flask import Flask, render_template, Response
from ultralytics import YOLO
import cv2
import numpy as np
from threading import Thread
import time

app = Flask(__name__)

# Global variables
camera = None
output_frame = None
lock = None

@app.route('/')
def index():
    return render_template('index.html')

def detect_signs():
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
        
        # Encode the frame in JPEG format
        with lock:
            output_frame = annotated_frame.copy()

def generate():
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
        time.sleep(0.1)  # Add small delay

@app.route('/video_feed')
def video_feed():
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def initialize_camera():
    global camera, lock
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera = cv2.VideoCapture(1)  # Try another camera index
    
    if not camera.isOpened():
        raise RuntimeError("Could not start camera.")

if __name__ == '__main__':
    from threading import Lock
    
    # Initialize global variables
    lock = Lock()
    initialize_camera()
    
    # Start a thread that will perform sign detection
    t = Thread(target=detect_signs, args=())
    t.daemon = True
    t.start()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)

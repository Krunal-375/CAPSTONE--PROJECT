from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = Flask(__name__)
app.template_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'templates'))

# Initialize the YOLO model
model = None

def init_model():
    global model
    if model is None:
        model = YOLO('best.pt')
    return model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"})

# For Vercel deployment
app.debug = False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

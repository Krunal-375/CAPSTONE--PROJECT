import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from threading import Thread
import time
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# Set page config
st.set_page_config(page_title="Sign Language Detection", layout="wide")

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

model = load_model()

# Title
st.title("Sign Language Detection")
st.write("This application detects sign language gestures in real-time using YOLOv8.")

def process_frame(frame):
    """Process frame through YOLOv8 model"""
    results = model(frame, stream=True)
    for result in results:
        # Convert tensor to numpy
        result_numpy = result.plot()
        return result_numpy
    return frame

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Process the frame
        processed_img = process_frame(img)
        
        return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

# Create a WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="sign-language",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Instructions
with st.expander("Instructions", expanded=True):
    st.markdown("""
    ### How to use:
    1. Allow camera access when prompted
    2. Position yourself in front of the camera
    3. Make sign language gestures
    4. The application will detect and display the signs in real-time
    
    ### Tips:
    - Ensure good lighting
    - Keep your gestures clear and within frame
    - Maintain a reasonable distance from the camera
    """)

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit and YOLOv8")

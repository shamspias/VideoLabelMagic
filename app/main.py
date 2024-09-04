import os
import streamlit as st
from config import Config
from extractor import VideoFrameExtractor

config = Config()

st.title(config.streamlit_title)

uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

# Filter for files ending with .pt
models = [file for file in os.listdir(config.models_directory) if file.endswith('.pt')]
model_selection = st.selectbox("Choose a model:", models)

output_dir = st.text_input("Output directory", config.output_directory)
frame_rate = st.number_input("Frame rate", value=config.default_frame_rate)

model_confidence = st.number_input("Model Confidence", value=0.1)

if st.button('Extract Frames'):
    if uploaded_file is not None:
        video_path = 'temp_video.mp4'
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        extractor = VideoFrameExtractor(video_path, frame_rate, output_dir, model_selection)
        extractor.extract_frames(model_confidence)
        st.success('Extraction Completed!')
    else:
        st.error("Please upload a file to proceed.")

import os
import uuid
import streamlit as st
from config import Config
from extractor import VideoFrameExtractor
from formats.roboflow_format import RoboflowFormat
from formats.cvat_format import CVATFormat

# Import other formats if available

config = Config()

st.title(config.streamlit_title)

uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])

# Update directory path for class configurations
class_config_files = [f for f in os.listdir(config.object_class_directory) if f.endswith('.yaml')]
class_config_selection = st.selectbox("Choose class configuration:", class_config_files)

# Filter for files ending with .pt
models = [file for file in os.listdir(config.models_directory) if file.endswith('.pt')]
model_selection = st.selectbox("Choose a model:", models)

output_dir = st.text_input("Output directory", config.output_directory)
frame_rate = st.number_input("Frame rate", value=config.default_frame_rate)
model_confidence = st.number_input("Model Confidence", value=0.1)

# Allow users to choose the output format
format_options = {'Roboflow': RoboflowFormat, 'CVAT': CVATFormat}  # Add more formats to this dictionary
format_selection = st.selectbox("Choose output format:", list(format_options.keys()))

if st.button('Extract Frames'):
    if uploaded_file is not None:
        # Create temp directory if it does not exist
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Generate a unique filename
        unique_filename = uploaded_file.name[:5] + "_" + str(uuid.uuid4())
        video_filename = unique_filename + ".mp4"
        video_path = os.path.join(temp_dir, video_filename)

        # Save the uploaded file
        with open(video_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())

        # Construct the class configuration path
        class_config_path = os.path.join(config.object_class_directory, class_config_selection)

        # Create a specific output directory named after the unique file
        specific_output_dir = os.path.join(output_dir, unique_filename)
        os.makedirs(specific_output_dir, exist_ok=True)

        # Instantiate the selected output format
        output_format_instance = format_options[format_selection](specific_output_dir)

        # Extract frames using the VideoFrameExtractor with the chosen format
        try:
            extractor = VideoFrameExtractor(video_path, frame_rate, specific_output_dir, model_selection,
                                            class_config_path, output_format_instance)
            extractor.extract_frames(model_confidence)
            st.success('Extraction Completed!')
            # Delete the temporary video file after successful extraction
            os.remove(video_path)
        except Exception as e:
            st.error(f"An error occurred during frame extraction: {str(e)}")
    else:
        st.error("Please upload a file to proceed.")

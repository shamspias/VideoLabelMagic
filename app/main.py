import os
import uuid
import streamlit as st
from config import Config
from extractor import VideoFrameExtractor
from formats.roboflow_format import RoboflowFormat
from formats.cvat_format import CVATFormat
from utils.storage_manager import StorageManager


class VideoLabelApp:
    def __init__(self):
        self.config = Config()
        self.storage_manager = StorageManager(self.config)
        self.format_options = {'Roboflow': RoboflowFormat, 'CVAT': CVATFormat}
        self.setup_ui()

    def setup_ui(self):
        """
        Set up the Streamlit user interface for VideoLabelMagic.
        """
        st.title(self.config.streamlit_title)
        st.sidebar.header("Storage Options")
        self.storage_option = st.sidebar.radio("Choose storage type:", ('Local', 'Object Storage'))

        if self.storage_option == 'Local':
            self.uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
            if self.uploaded_file is not None:
                self.process_ui()

    def process_ui(self):
        """
        Continue UI setup based on user input and file selection.
        """
        class_config_files = [f for f in os.listdir(self.config.object_class_directory) if f.endswith('.yaml')]
        self.class_config_selection = st.selectbox("Choose class configuration:", class_config_files)
        models = [file for file in os.listdir(self.config.models_directory) if file.endswith('.pt')]
        self.model_selection = st.selectbox("Choose a model:", models)
        self.frame_rate = st.number_input("Frame rate", value=self.config.default_frame_rate)
        self.model_confidence = st.number_input("Model Confidence", value=0.1)
        transformation_options = st.multiselect('Select image transformations:',
                                                ['Resize', 'Grayscale', 'Rotate 90 degrees'])
        self.transformations = {
            'resize': 'Resize' in transformation_options,
            'grayscale': 'Grayscale' in transformation_options,
            'rotate': 'Rotate 90 degrees' in transformation_options
        }
        self.format_selection = st.selectbox("Choose output format:", list(self.format_options.keys()))
        if st.button('Extract Frames'):
            self.process_video()

    def process_video(self):
        """
        Process the uploaded video file according to the specified settings and store outputs in the selected storage type.
        """
        if self.uploaded_file is not None:
            self.process_local_video()

    def process_local_video(self):
        """
        Handle local video processing, saving the uploaded file temporarily before processing.
        """
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        unique_filename = self.uploaded_file.name[:5] + "_" + str(uuid.uuid4())
        video_filename = unique_filename + ".mp4"
        video_path = os.path.join(temp_dir, video_filename)

        with open(video_path, 'wb') as f:
            f.write(self.uploaded_file.getbuffer())

        # Check if the file was written and exists before proceeding
        if os.path.exists(video_path) and os.path.isfile(video_path):
            st.write(f"Debug: File path is {video_path}")  # Debugging output
            self.run_extraction(video_path, unique_filename)
        else:
            st.error("Failed to save the file locally or file path is incorrect.")

    def run_extraction(self, video_path, unique_filename):
        """
        Run the extraction process using the specified VideoFrameExtractor.
        """
        class_config_path = os.path.join(self.config.object_class_directory, self.class_config_selection)
        specific_output_dir = os.path.join(self.config.output_directory, unique_filename)
        os.makedirs(specific_output_dir, exist_ok=True)
        output_format_instance = self.format_options[self.format_selection](specific_output_dir)

        try:
            extractor = VideoFrameExtractor(self.config, video_path, self.frame_rate, specific_output_dir,
                                            self.model_selection, class_config_path, output_format_instance,
                                            self.transformations)
            extractor.extract_frames(self.model_confidence)
            if self.format_selection == "CVAT":
                output_format_instance.zip_and_cleanup()
            st.success('Extraction Completed!')
            os.remove(video_path)
        except Exception as e:
            st.error(f"An error occurred during frame extraction: {str(e)}")


if __name__ == "__main__":
    app = VideoLabelApp()

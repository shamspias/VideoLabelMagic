import os
import uuid
import shutil
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
        st.title(self.config.streamlit_title)
        st.sidebar.header("Storage Options")
        self.storage_option = st.sidebar.radio("Choose storage type:", ('Local', 'Object Storage'))

        if self.storage_option == 'Object Storage':
            self.handle_object_storage()
        elif self.storage_option == 'Local':
            self.handle_local_storage()

    def handle_object_storage(self):
        if not self.config.storage_use_s3:
            st.sidebar.error("Object storage is not configured properly in .env file.")
            return
        files = self.storage_manager.list_files_in_bucket()
        self.selected_file = st.selectbox("Select a file from Object Storage:", files)
        self.continue_ui()

    def handle_local_storage(self):
        self.uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'avi', 'mov'])
        self.continue_ui()

    def continue_ui(self):
        class_config_files = [f for f in os.listdir(self.config.object_class_directory) if f.endswith('.yaml')]
        self.class_config_selection = st.selectbox("Choose class configuration:", class_config_files)
        models = [file for file in os.listdir(self.config.models_directory) if file.endswith('.pt')]
        self.model_selection = st.selectbox("Choose a model:", models)
        self.frame_rate = st.number_input("Frame rate", value=self.config.default_frame_rate)
        transformation_options = st.multiselect('Select image transformations:',
                                                ['Resize', 'Grayscale', 'Rotate 90 degrees'])
        self.transformations = {
            'resize': 'Resize' in transformation_options,
            'grayscale': 'Grayscale' in transformation_options,
            'rotate': 'Rotate 90 degrees' in transformation_options
        }
        self.format_selection = st.selectbox("Choose output format:", list(self.format_options.keys()))
        self.sahi_enabled = st.sidebar.checkbox("Enable SAHI", value=self.config.sahi_enabled)
        if self.sahi_enabled:
            self.config.sahi_model_type = st.sidebar.selectbox("Model Architecture:", ["yolov8", "yolov9", "yolov10"])
            self.config.sahi_slice_size = st.sidebar.slider("SAHI slice size:", 128, 512, (256, 256))
            self.config.sahi_overlap_ratio = st.sidebar.slider("SAHI overlap ratio:", 0.1, 0.5, 0.2)
        self.model_confidence = st.number_input("Model Confidence", value=0.1)

        if st.button('Extract Frames'):
            self.process_video()

    def process_video(self):
        if self.storage_option == 'Local' and self.uploaded_file is not None:
            self.process_local_video()
        elif self.storage_option == 'Object Storage' and self.selected_file:
            self.process_cloud_storage_video()

    def process_local_video(self):
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)
        unique_filename = self.uploaded_file.name[:5] + "_" + str(uuid.uuid4())
        video_filename = unique_filename + ".mp4"
        video_path = os.path.join(temp_dir, video_filename)
        with open(video_path, 'wb') as f:
            f.write(self.uploaded_file.getbuffer())
        self.run_extraction(video_path, unique_filename)

    def process_cloud_storage_video(self):
        """
        Handle the download of the file from cloud storage, rename it similar to the local process,
        and perform the frame extraction.
        """
        temp_dir = 'temp'
        os.makedirs(temp_dir, exist_ok=True)

        # Generate unique filename similar to the local upload handling
        file_basename = os.path.basename(self.selected_file)
        unique_filename = file_basename[:5] + "_" + str(uuid.uuid4()) + ".mp4"  # Consistent renaming
        video_path = os.path.join(temp_dir, unique_filename)

        # Download the file from S3 into the temp directory
        self.storage_manager.download_file_from_s3(self.selected_file, video_path)

        # Proceed to run the extraction process
        self.run_extraction(video_path, unique_filename)

    def run_extraction(self, video_path, unique_filename):
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
            if self.storage_option == 'Object Storage':
                self.upload_outputs(specific_output_dir)

            # Clean up: Remove the temporary video file after processing
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted temporary video file: {video_path}")

            st.success('Extraction Completed!')
        except Exception as e:
            st.error(f"An error occurred during frame extraction: {str(e)}")

    def upload_outputs(self, directory):
        """
        Upload all files and directories from the specified directory to the S3 bucket,
        maintaining the same structure under a 'processed/' prefix in S3.
        Args:
            directory (str): The local directory path containing the files to be uploaded.
        """
        # Determine the base path for the directory to maintain structure in S3
        base_path = os.path.dirname(directory)

        # Walk through the directory and upload each file to S3
        for root, dirs, files in os.walk(directory):
            for file in files:
                local_file_path = os.path.join(root, file)
                # Calculate the relative path for S3 key to maintain the folder structure
                relative_path = os.path.relpath(local_file_path, base_path)
                s3_object_name = os.path.join("processed", relative_path)  # Use 'processed/' prefix in S3

                # Upload the file to S3, preserving directory structure
                self.storage_manager.upload_file_to_s3(local_file_path, s3_object_name)
                print(f"Uploaded {local_file_path} to S3 as {s3_object_name}")

        # Optionally, delete the directory locally after uploading
        shutil.rmtree(directory)
        print(f"Deleted local directory after upload: {directory}")


if __name__ == "__main__":
    app = VideoLabelApp()

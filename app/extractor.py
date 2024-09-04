import cv2
import os
from ultralytics import YOLO
import yaml


class VideoFrameExtractor:
    """
    A class to extract frames from video at specified intervals and annotate them using YOLO model predictions.

    Attributes:
        video_path (str): Path to the video file.
        frame_rate (float): Desired frame rate to extract images.
        output_dir (str): Base directory to store extracted images and annotations.
        model_path (str): Path to the YOLO model for object detection.
        class_config_path (str): Path to the class configuration file.
        output_format (object): Format handler for saving annotations.
    """

    def __init__(self, video_path, frame_rate, output_dir, model_path, class_config_path, output_format):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.yolo_model = YOLO(os.path.join('models', model_path))
        self.output_format = output_format
        self.supported_classes = self.load_classes(class_config_path)

    def load_classes(self, config_path):
        """
        Loads object classes from a YAML configuration file.
        """
        with open(config_path, 'r') as file:
            class_data = yaml.safe_load(file)
        return [cls['name'] for cls in class_data['classes']]

    def extract_frames(self, model_confidence):
        """
        Extracts frames from the video file at the specified frame rate, annotates them using the YOLO model,
        and saves using the specified format.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.frame_rate))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = f"{self._get_video_basename()}_image{frame_count}.jpg"
                frame_path = os.path.join(self.output_dir, 'images', frame_filename)
                cv2.imwrite(frame_path, frame)
                results = self.yolo_model.predict(frame, conf=model_confidence)
                self.output_format.save_annotations(frame, frame_path, frame_filename, results, self.supported_classes)

            frame_count += 1

        cap.release()

    def _get_video_basename(self):
        """
        Extracts the basename of the video file without its extension.
        """
        return os.path.splitext(os.path.basename(self.video_path))[0]

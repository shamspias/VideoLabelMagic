import cv2
import os
from ultralytics import YOLO
import yaml
from utils.image_processor import ImageProcessor
from utils.sahi_utils import SahiUtils


class VideoFrameExtractor:
    """
    Extracts frames from video at specified intervals, applies selected transformations,
    and annotates them using YOLO model predictions, with options to save locally or to object storage.
    """

    def __init__(self, config, video_path, frame_rate, output_dir, model_path, class_config_path, output_format,
                 transformations):
        self.config = config
        self.video_path = video_path  # Ensure this is a string representing the path to the video file.
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.yolo_model = YOLO(os.path.join('models', model_path))
        self.class_config_path = class_config_path
        self.output_format = output_format
        self.transformations = transformations
        self.supported_classes = self.load_classes(self.class_config_path)
        self.image_processor = ImageProcessor(output_size=self.transformations.get('size', (640, 640)))

        # Debugging output to ensure path handling
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"The specified video file was not found at {self.video_path}")
        else:
            print(f"VideoFrameExtractor initialized with video path: {self.video_path}")

    def load_classes(self, config_path):
        """
        Load classes from a YAML configuration file.
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        with open(config_path, 'r') as file:
            class_data = yaml.safe_load(file)
        return [cls['name'] for cls in class_data['classes']]

    def extract_frames(self, model_confidence):
        """
        Extract and process frames from the video, and save them using the specified output format.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video stream for {self.video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(video_fps / self.frame_rate))
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                transformed_images = self.apply_transformations(frame)

                for key, transformed_image in transformed_images.items():
                    if transformed_image.ndim == 2:  # Check if the image is grayscale
                        transformed_image = cv2.cvtColor(transformed_image,
                                                         cv2.COLOR_GRAY2BGR)  # Convert back to RGB format for consistency

                    frame_filename = f"{self._get_video_basename()}_image{frame_count}_{key}.jpg"
                    frame_path = os.path.join(self.output_dir, 'images', frame_filename)

                    # Save images locally or to configured storage
                    cv2.imwrite(frame_path, transformed_image)
                    results = self.yolo_model.predict(transformed_image, conf=model_confidence)
                    self.output_format.save_annotations(transformed_image, frame_path, frame_filename, results,
                                                        self.supported_classes)

            frame_count += 1

        cap.release()

    def apply_transformations(self, frame):
        """
        Apply selected transformations to the frame and return a dictionary of transformed images.
        """
        transformed_images = {}
        if 'resize' in self.transformations and self.transformations['resize']:
            frame = self.image_processor.resize_image(frame)
            transformed_images['resized'] = frame

        if 'grayscale' in self.transformations and self.transformations['grayscale']:
            grayscale_image = self.image_processor.convert_to_grayscale(frame)
            transformed_images['grayscale'] = grayscale_image

        if 'rotate' in self.transformations and self.transformations['rotate']:
            rotated_image = self.image_processor.rotate_image_90_degrees(frame)
            transformed_images['rotated'] = rotated_image

        if not transformed_images:
            transformed_images['original'] = frame

        return transformed_images

    def _get_video_basename(self):
        """
        Extract the basename of the video file without extension.
        """
        return os.path.splitext(os.path.basename(self.video_path))[0]

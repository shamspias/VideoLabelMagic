import cv2
import os
from ultralytics import YOLO
import yaml
from utils.image_processor import ImageProcessor  # Make sure this is the correct import path


class VideoFrameExtractor:
    """
    Extracts frames from video at specified intervals, applies selected transformations,
    and annotates them using YOLO model predictions.
    """

    def __init__(self, video_path, frame_rate, output_dir, model_path, class_config_path, output_format,
                 transformations):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.output_dir = output_dir
        self.yolo_model = YOLO(os.path.join('models', model_path))
        self.output_format = output_format
        self.supported_classes = self.load_classes(class_config_path)
        self.transformations = transformations
        self.image_processor = ImageProcessor(output_size=self.transformations.get('size', (640, 640)))

    def load_classes(self, config_path):
        with open(config_path, 'r') as file:
            class_data = yaml.safe_load(file)
        return [cls['name'] for cls in class_data['classes']]

    def extract_frames(self, model_confidence):
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
                transformed_images = self.apply_transformations(frame)

                for key, transformed_image in transformed_images.items():
                    if transformed_image.ndim == 2:  # Check if the image is grayscale
                        transformed_image = cv2.cvtColor(transformed_image,
                                                         cv2.COLOR_GRAY2BGR)  # Convert back to RGB format for consistency

                    frame_filename = f"{self._get_video_basename()}_image{frame_count}_{key}.jpg"
                    frame_path = os.path.join(self.output_dir, 'images', frame_filename)
                    cv2.imwrite(frame_path, transformed_image)
                    results = self.yolo_model.predict(transformed_image, conf=model_confidence)
                    self.output_format.save_annotations(transformed_image, frame_path, frame_filename, results,
                                                        self.supported_classes)

            frame_count += 1

        cap.release()

    def apply_transformations(self, frame):
        # Dictionary to hold transformed images
        transformed_images = {}

        # Apply resizing if selected
        if 'resize' in self.transformations and self.transformations['resize']:
            frame = self.image_processor.resize_image(frame)
            transformed_images['resized'] = frame  # Store resized image

        # Apply grayscale transformation if selected
        if 'grayscale' in self.transformations and self.transformations['grayscale']:
            grayscale_image = self.image_processor.convert_to_grayscale(frame)
            transformed_images['grayscale'] = grayscale_image  # Store grayscale image

        # Apply 90-degree rotation if selected
        if 'rotate' in self.transformations and self.transformations['rotate']:
            rotated_image = self.image_processor.rotate_image_90_degrees(frame)
            transformed_images['rotated'] = rotated_image  # Store rotated image

        # If no transformations are selected, add the original image
        if not transformed_images:
            transformed_images['original'] = frame

        return transformed_images

    def _get_video_basename(self):
        return os.path.splitext(os.path.basename(self.video_path))[0]

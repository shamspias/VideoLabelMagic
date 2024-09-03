import cv2
import os
from ultralytics import YOLO
import yaml


class VideoFrameExtractor:
    def __init__(self, video_path, frame_rate, output_dir, model_path):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.output_dir = os.path.join(output_dir, 'train')
        self.image_dir = os.path.join(self.output_dir, 'images')
        self.label_dir = os.path.join(self.output_dir, 'labels')
        self.yolo_model = YOLO(os.path.join('models', model_path))
        self.supported_classes = ['person', 'car', 'truck', 'tank']

        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        self._create_data_yaml()

    def _create_data_yaml(self):
        data = {
            'train': os.path.abspath(self.image_dir),
            'nc': len(self.supported_classes),
            'names': self.supported_classes
        }
        with open(os.path.join(self.output_dir, '..', 'data.yaml'), 'w') as file:
            yaml.dump(data, file)

    def extract_frames(self):
        cap = cv2.VideoCapture(self.video_path)
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(video_fps / self.frame_rate)

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_filename = f"{frame_count}.jpg"
                frame_path = os.path.join(self.image_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                # Annotation logic here (placeholder)

            frame_count += 1
        cap.release()

    # Additional methods for annotation, etc.

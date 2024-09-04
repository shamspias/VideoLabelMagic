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
    """

    def __init__(self, video_path, frame_rate, output_dir, model_path):
        self.video_path = video_path
        self.frame_rate = frame_rate
        self.output_dir = os.path.join(output_dir, 'train')
        self.image_dir = os.path.join(self.output_dir, 'images')
        self.label_dir = os.path.join(self.output_dir, 'labels')
        self.yolo_model = YOLO(os.path.join('models', model_path))
        self.supported_classes = ['person', 'car', 'truck', 'tank']

        # Ensure necessary directories exist
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

        # Create metadata for training
        self._create_data_yaml()

    def _create_data_yaml(self):
        """
        Creates a YAML file to store metadata about the training dataset.
        """
        data = {
            'train': os.path.abspath(self.image_dir),
            'nc': len(self.supported_classes),
            'names': self.supported_classes
        }
        with open(os.path.join(self.output_dir, '..', 'data.yaml'), 'w') as file:
            yaml.dump(data, file)

    def extract_frames(self, model_confidence):
        """
        Extracts frames from the video file at the specified frame rate and saves them in the image directory.

        model_confidence (float): Model confidence that help to annotated
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
                frame_path = os.path.join(self.image_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                self._annotate_frame(frame, frame_path, frame_filename, model_confidence)

            frame_count += 1

        cap.release()

    def _annotate_frame(self, frame, frame_path, frame_filename, model_conf):
        """
        Annotates the frame using the YOLO model and saves the annotation to a file.

        Parameters:
            frame (np.array): The frame to be annotated.
            frame_path (str): Path where the frame image is saved.
            frame_filename (str): Filename of the frame image.
        """
        results = self.yolo_model.predict(frame, conf=model_conf)
        annotation_filename = frame_filename.replace('.jpg', '.txt')
        annotation_path = os.path.join(self.label_dir, annotation_filename)
        img_height, img_width = frame.shape[:2]

        with open(annotation_path, 'w') as f:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        if self.supported_classes[class_id] in self.supported_classes:
                            confidence = box.conf[0]
                            xmin, ymin, xmax, ymax = box.xyxy[0]
                            x_center = ((xmin + xmax) / 2) / img_width
                            y_center = ((ymin + ymax) / 2) / img_height
                            width = (xmax - xmin) / img_width
                            height = (ymax - ymin) / img_height
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    def _get_video_basename(self):
        """
        Extracts the basename of the video file without its extension.

        Returns:
            str: The basename of the video file.
        """
        return os.path.splitext(os.path.basename(self.video_path))[0]

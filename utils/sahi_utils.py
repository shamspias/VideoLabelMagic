import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
import numpy as np
import matplotlib.pyplot as plt


class SahiUtils:
    def __init__(self, debug, model_path, model_type='yolov8', device='cpu', slice_size=(256, 256),
                 overlap_ratio=(0.2, 0.2)):
        self.debug = debug
        self.device = device  # Can be 'cpu' or 'cuda:0' for GPU
        self.model_type = model_type
        self.model = self.load_model(model_path)
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio

    def load_model(self, model_path):
        """Loads a detection model based on the specified type and path."""
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=self.model_type,
            model_path=model_path,
            confidence_threshold=0.1,
            device=self.device,
        )
        return detection_model

    def show_image(self, image, title="Image"):
        """Displays a NumPy image using matplotlib."""
        # Convert BGR to RGB for correct color
        plt.imshow(image if len(image.shape) == 2 else cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        ))
        plt.title(title)
        plt.axis('off')  # Hide axes
        plt.show()

    def perform_sliced_inference(self, image):
        """Performs object detection on an image using sliced prediction."""
        pil_image = read_image_as_pil(image)
        results = get_sliced_prediction(
            pil_image,
            detection_model=self.model,
            slice_height=self.slice_size[0],
            slice_width=self.slice_size[1],
            overlap_height_ratio=self.overlap_ratio[0],
            overlap_width_ratio=self.overlap_ratio[1],
            verbose=False
        )
        if self.debug:
            self.show_image(image)

        return self.format_predictions(results)

    def format_predictions(self, prediction_result):
        """Formats the predictions into a compatible format with YOLO output."""
        formatted_results = {'boxes': []}
        for prediction in prediction_result.object_prediction_list:
            box = prediction.bbox.to_voc_bbox()
            formatted_result = {
                'cls': [prediction.category.id],  # list wrapping for compatibility
                'conf': [prediction.score.value],  # list wrapping for compatibility
                'xyxy': [np.array([box[0], box[1], box[2], box[3]])],  # VOC format to numpy array
            }
            formatted_results['boxes'].append(formatted_result)

        return formatted_results

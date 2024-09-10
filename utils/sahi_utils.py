import cv2
import uuid
import time
import os
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import read_image_as_pil
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class SahiUtils:
    def __init__(self, debug,
                 supported_classes_map,
                 model_path,
                 model_type='yolov8',
                 device='cpu',
                 slice_size=(256, 256),
                 overlap_ratio=(0.2, 0.2)):
        self.debug = debug
        self.supported_classes_map = supported_classes_map
        self.device = device  # Can be 'cpu' or 'cuda:0' for GPU
        self.model_type = model_type
        self.model = self.load_model(model_path)
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.debug_annotated_directory = str(uuid.uuid4())

    def load_model(self, model_path):
        """Loads a detection model based on the specified type and path."""
        # print(self.supported_classes_map)
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=self.model_type,
            model=model_path,
            confidence_threshold=0.1,
            device=self.device,
            # category_mapping=self.supported_classes_map,
            # category_remapping=self.supported_classes_map,
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

    def show_annotated_image(self, image_path):
        img = Image.open(image_path)
        plt.imshow(img)
        plt.axis('off')
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
            postprocess_class_agnostic=True,
            verbose=False
        )
        if self.debug:
            random_value = str(uuid.uuid4())
            # Start exporting the image
            results.export_visuals(export_dir=f"temp/{self.debug_annotated_directory}/", file_name=random_value)

            # Wait until the file is created
            file_path = f"temp/{self.debug_annotated_directory}/{random_value}.png"
            timeout = 10  # Set a timeout limit of 10 seconds or more if necessary
            start_time = time.time()

            while not os.path.exists(file_path):
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"File creation exceeded {timeout} seconds.")
                time.sleep(0.1)  # Wait for 100 milliseconds before checking again

            # Once the file exists, display it
            self.show_annotated_image(file_path)

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

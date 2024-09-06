from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
import numpy as np


class SahiUtils:
    def __init__(self, model_path, model_type='yolov8', device='cpu', slice_size=(256, 256), overlap_ratio=(0.2, 0.2)):
        self.device = device  # CPU or 'cuda:0'
        self.model_type = model_type
        self.model = self.load_model(model_path)
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio

    def load_model(self, model_path):
        detection_model = AutoDetectionModel.from_pretrained(
            model_type=self.model_type,
            model_path=model_path,
            confidence_threshold=0.1,
            device=self.device,
        )
        return detection_model

    def perform_sliced_inference(self, image):
        results = get_sliced_prediction(
            image,
            self.model,  # this should be a sahi model
            slice_height=self.slice_size[0],
            slice_width=self.slice_size[1],
            overlap_height_ratio=self.overlap_ratio[0],
            overlap_width_ratio=self.overlap_ratio[1],
            verbose=False
        )
        return self.format_predictions(results, image)

    def format_predictions(self, prediction_result, image):
        formatted_results = {"boxes": [], "names": {}, "orig_img": image, "orig_shape": image.shape, "path": "",
                             "probs": None, "save_dir": None, "speed": None}
        class_ids = set()
        for prediction in prediction_result.object_prediction_list:
            class_id = prediction.category.id
            class_ids.add(class_id)
            formatted_results["names"][class_id] = prediction.category.name
            bbox_xyxy = [prediction.bbox.minx, prediction.bbox.miny, prediction.bbox.maxx, prediction.bbox.maxy]
            formatted_results["boxes"].append({
                "class_id": class_id,
                "bbox": np.array(bbox_xyxy),
                "score": prediction.score.value
            })

        formatted_results["boxes"] = self.convert_boxes(formatted_results["boxes"])
        return formatted_results

    def convert_boxes(self, boxes):
        # Convert to ultralytics.engine.results.Boxes format or similar
        # Ensure correct shape and concatenation of score and class_id
        boxes_array = [np.concatenate([box["bbox"], [box["score"], box["class_id"]]]) for box in boxes]
        if boxes_array:  # Check if list is not empty
            return np.stack(boxes_array)  # Properly stack arrays to maintain structure
        else:
            return np.array([])  # Return an empty numpy array if no boxes

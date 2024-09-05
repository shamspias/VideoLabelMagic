from sahi.predict import get_sliced_prediction
from ultralytics import YOLO


class SahiUtils:
    def __init__(self, model_path, device='cpu', slice_size=(256, 256), overlap_ratio=(0.2, 0.2)):
        self.device = device
        self.model = self.load_model(model_path)
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio

    def load_model(self, model_path):
        # Load YOLO model using Ultralytics methods
        model = YOLO(model_path)
        model.to(self.device)
        return model

    def perform_sliced_inference(self, image):
        # Perform sliced inference using the loaded model and SAHI
        results = get_sliced_prediction(
            image=image,
            detection_model=self.model,
            slice_height=self.slice_size[0],
            slice_width=self.slice_size[1],
            overlap_height_ratio=self.overlap_ratio[0],
            overlap_width_ratio=self.overlap_ratio[1]
        )
        return results

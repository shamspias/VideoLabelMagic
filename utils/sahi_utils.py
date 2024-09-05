from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel


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
        # Perform sliced inference using the loaded model and SAHI
        results = get_sliced_prediction(
            image,
            self.model,  # this should be a sahi model
            slice_height=self.slice_size[0],
            slice_width=self.slice_size[1],
            overlap_height_ratio=self.overlap_ratio[0],
            overlap_width_ratio=self.overlap_ratio[1]
        )
        return results

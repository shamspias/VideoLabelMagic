from formats.base_format import BaseFormat
import os
import yaml


class RoboflowFormat(BaseFormat):
    def __init__(self, output_dir, sahi_enabled):
        super().__init__(output_dir, sahi_enabled)
        self.image_dir = os.path.join(output_dir, 'images')
        self.label_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def save_annotations(self, frame, frame_path, frame_filename, results, supported_classes):
        img_dimensions = frame.shape[:2]
        annotations = self.process_results(frame, results, img_dimensions)
        self.write_annotations(frame_filename, annotations)
        self.create_data_yaml(supported_classes)

    def create_data_yaml(self, supported_classes):
        """
        Creates a YAML file to store metadata about the training dataset.
        """
        data = {
            'train': os.path.abspath(self.image_dir),
            'nc': len(supported_classes),
            'names': supported_classes
        }
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as file:
            yaml.dump(data, file)

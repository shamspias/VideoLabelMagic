from formats.base_format import BaseFormat
import os
import yaml


class RoboflowFormat(BaseFormat):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.image_dir = os.path.join(self.output_dir, 'train', 'images')
        self.label_dir = os.path.join(self.output_dir, 'train', 'labels')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def save_annotations(self, annotations, class_names):
        # Generate data.yaml
        data_yaml = {
            'train': os.path.abspath(self.image_dir),
            'nc': len(class_names),
            'names': class_names
        }
        with open(os.path.join(self.output_dir, 'data.yaml'), 'w') as file:
            yaml.dump(data_yaml, file)

        # Save individual annotation files
        for annotation in annotations:
            with open(os.path.join(self.label_dir, annotation['filename']), 'w') as f:
                for obj in annotation['objects']:
                    f.write(f"{obj['class_id']} {obj['x_center']} {obj['y_center']} {obj['width']} {obj['height']}\n")

from formats.base_format import BaseFormat
import os
import yaml


class RoboflowFormat(BaseFormat):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.image_dir = os.path.join(output_dir, 'images')
        self.label_dir = os.path.join(output_dir, 'labels')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def save_annotations(self, frame, frame_path, frame_filename, results, supported_classes):
        """
        Saves the annotations in the Roboflow specified format.
        """
        annotation_filename = frame_filename.replace('.jpg', '.txt')
        annotation_path = os.path.join(self.label_dir, annotation_filename)
        img_height, img_width = frame.shape[:2]

        with open(annotation_path, 'w') as f:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        if supported_classes[class_id] in supported_classes:
                            confidence = box.conf[0]
                            xmin, ymin, xmax, ymax = box.xyxy[0]
                            x_center = ((xmin + xmax) / 2) / img_width
                            y_center = ((ymin + ymax) / 2) / img_height
                            width = (xmax - xmin) / img_width
                            height = (ymax - ymin) / img_height
                            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

        # Generate metadata file if needed
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

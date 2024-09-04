from formats.base_format import BaseFormat
import os


class CVATFormat(BaseFormat):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.image_dir = os.path.join(output_dir, 'train', 'images')
        self.label_dir = os.path.join(output_dir, 'train', 'labels')
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)

    def save_annotations(self, frame, frame_path, frame_filename, results, supported_classes):
        annotation_filename = frame_filename.replace('.jpg', '.txt')
        annotation_path = os.path.join(self.label_dir, annotation_filename)
        img_height, img_width = frame.shape[:2]

        with open(annotation_path, 'w') as f:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = box.conf[0]
                        xmin, ymin, xmax, ymax = box.xyxy[0]
                        f.write(f"{class_id} {xmin:.6f} {ymin:.6f} {xmax:.6f} {ymax:.6f} {confidence:.2f}\n")

    def create_metadata_files(self, supported_classes):
        obj_names_path = os.path.join(self.output_dir, 'obj.names')
        obj_data_path = os.path.join(self.output_dir, 'obj.data')
        train_txt_path = os.path.join(self.output_dir, 'train.txt')

        with open(obj_names_path, 'w') as f:
            f.writelines(f"{cls}\n" for cls in supported_classes)

        with open(obj_data_path, 'w') as f:
            f.writelines([
                f"classes = {len(supported_classes)}\n",
                f"train = {os.path.abspath(self.image_dir)}\n",
                f"valid = {os.path.abspath(self.image_dir)}\n",
                f"names = {obj_names_path}\n",
                f"backup = backup/\n"
            ])

        with open(train_txt_path, 'w') as f:
            for image_file in os.listdir(self.image_dir):
                f.write(f"{os.path.join(os.path.abspath(self.image_dir), image_file)}\n")

    def ensure_directories(self):
        """ Ensures that necessary directories are created for CVAT. """
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.label_dir, exist_ok=True)
        # Assuming supported_classes is globally accessible or passed to this method when called.
        self.create_metadata_files(self.supported_classes)  # Corrected to use an attribute

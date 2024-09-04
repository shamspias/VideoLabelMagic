import os
import cv2
from formats.base_format import BaseFormat


class CVATFormat(BaseFormat):
    """
    Class to handle the CVAT format for image annotations.

    Attributes:
        output_dir (str): Base directory for all output.
    """

    def __init__(self, output_dir):
        """
        Initialize the CVATFormat with specific directories for images and annotations.

        Args:
            output_dir (str): The base directory for all outputs.
        """
        super().__init__(output_dir)
        self.data_dir = os.path.join(output_dir, 'data')
        self.image_dir = os.path.join(self.data_dir, 'obj_train_data')
        os.makedirs(self.image_dir, exist_ok=True)

    def save_annotations(self, frame, frame_path, frame_filename, results, supported_classes):
        frame_filename = frame_filename.replace('.jpg', '.png')  # Ensuring PNG format for CVAT
        image_path = os.path.join(self.image_dir, frame_filename)
        cv2.imwrite(image_path, frame)  # Save the frame image

        annotation_filename = frame_filename.replace('.png', '.txt')
        annotation_path = os.path.join(self.image_dir, annotation_filename)

        with open(annotation_path, 'w') as file:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        # Ensure that you're handling tensor dimensions correctly
                        if box.xyxy.dim() == 2 and box.xyxy.shape[0] == 1:  # Assuming result.boxes.xyxy is a tensor
                            class_id = int(box.cls[0])  # class_id is extracted from a tensor
                            bbox = box.xyxy[0].tolist()  # Converting tensor to list
                            confidence = box.conf[0]
                            file.write(
                                f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {confidence:.2f}\n")

    def create_metadata_files(self, supported_classes):
        """
        Generates metadata files required for CVAT training configuration.
        """
        obj_names_path = os.path.join(self.data_dir, 'obj.names')
        obj_data_path = os.path.join(self.data_dir, 'obj.data')
        train_txt_path = os.path.join(self.data_dir, 'train.txt')

        with open(obj_names_path, 'w') as f:
            f.writelines(f"{cls}\n" for cls in supported_classes)

        with open(obj_data_path, 'w') as f:
            f.write("classes = {}\n".format(len(supported_classes)))
            f.write("train = {}\n".format(os.path.abspath(self.image_dir)))
            f.write("valid = {}\n".format(os.path.abspath(self.image_dir)))
            f.write("names = {}\n".format(obj_names_path))
            f.write("backup = backup/\n")

        with open(train_txt_path, 'w') as f:
            for image_file in os.listdir(self.image_dir):
                if image_file.endswith('.png'):
                    f.write(f"{os.path.join(self.image_dir, image_file)}\n")

    def ensure_directories(self):
        """
        Ensures all directories are created and metadata files are prepared.
        """
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        # Ensure supported_classes are defined or passed to this method when called
        self.create_metadata_files(self.supported_classes)

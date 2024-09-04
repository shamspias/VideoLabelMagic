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
        super().__init__(output_dir)
        self.data_dir = os.path.join(output_dir, 'data')
        self.image_dir = os.path.join(self.data_dir, 'obj_train_data')
        os.makedirs(self.image_dir, exist_ok=True)

    def save_annotations(self, frame, frame_path, frame_filename, results, supported_classes):
        """
        Saves annotations and images in CVAT-compatible format directly in obj_train_data.
        """
        # Convert to PNG for image file
        frame_filename_png = frame_filename.replace('.jpg', '.png')
        image_path = os.path.join(self.image_dir, frame_filename_png)
        cv2.imwrite(image_path, frame)  # Save the frame image

        # Text file for annotations stored in the same directory as images
        annotation_filename = frame_filename_png.replace('.png', '.txt')
        annotation_path = os.path.join(self.image_dir, annotation_filename)

        with open(annotation_path, 'w') as file:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if box.xyxy.dim() == 2 and box.xyxy.shape[0] == 1:
                            class_id = int(box.cls[0])
                            bbox = box.xyxy[0].tolist()
                            confidence = box.conf[0]
                            file.write(
                                f"{class_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {confidence:.2f}\n")

        # After saving all annotations, update metadata files
        self.create_metadata_files(supported_classes)

    def create_metadata_files(self, supported_classes):
        """
        Creates necessary metadata files for CVAT training setup.
        """
        obj_names_path = os.path.join(self.data_dir, 'obj.names')
        obj_data_path = os.path.join(self.data_dir, 'obj.data')
        train_txt_path = os.path.join(self.data_dir, 'train.txt')

        # Create obj.names file
        with open(obj_names_path, 'w') as f:
            for cls in supported_classes:
                f.write(f"{cls}\n")

        # Create obj.data file
        with open(obj_data_path, 'w') as f:
            f.write("classes = {}\n".format(len(supported_classes)))
            f.write("train = data/train.txt\n")
            f.write("names = data/obj.names\n")
            f.write("backup = backup/\n")

        # Create train.txt file listing all image files
        with open(train_txt_path, 'w') as f:
            for image_file in os.listdir(self.image_dir):
                if image_file.endswith('.png'):
                    f.write(f"data/obj_train_data/{image_file}\n")

    def ensure_directories(self):
        """Ensures all directories are created and ready for use."""
        super().ensure_directories()  # Ensures base directories are created

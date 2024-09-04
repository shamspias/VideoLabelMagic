import os
import cv2
import zipfile
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
        frame_filename_png = frame_filename.replace('.jpg', '.png')
        image_path = os.path.join(self.image_dir, frame_filename_png)
        cv2.imwrite(image_path, frame)  # Save the frame image

        annotation_filename = frame_filename_png.replace('.png', '.txt')
        annotation_path = os.path.join(self.image_dir, annotation_filename)

        with open(annotation_path, 'w') as file:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        if box.xyxy.dim() == 2 and box.xyxy.shape[0] == 1:
                            class_id = int(box.cls[0])
                            xmin, ymin, xmax, ymax = box.xyxy[0].tolist()
                            x_center = ((xmin + xmax) / 2) / frame.shape[1]
                            y_center = ((ymin + ymax) / 2) / frame.shape[0]
                            width = (xmax - xmin) / frame.shape[1]
                            height = (ymax - ymin) / frame.shape[0]
                            file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

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

    def zip_and_cleanup(self):
        # Create a zip file and add all the data in the data directory to it.
        zip_path = os.path.join(self.output_dir, 'cvat_data.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.data_dir, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, self.data_dir))
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    zipf.write(dir_path, os.path.relpath(dir_path, self.data_dir))

        # Clean up the directory by removing all files first, then empty directories.
        for root, dirs, files in os.walk(self.data_dir, topdown=False):
            for file in files:
                os.remove(os.path.join(root, file))
            for dir in dirs:
                os.rmdir(os.path.join(root, dir))

        # Finally, remove the base data directory now that it should be empty.
        os.rmdir(self.data_dir)

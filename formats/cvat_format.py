import os
import cv2
import zipfile
from typing import List
from formats.base_format import BaseFormat


class CVATFormat(BaseFormat):
    """
    Handles the CVAT format for image annotations. This class manages the creation of necessary directories,
    the writing of annotations into CVAT-compatible text files, and the organization of image data.
    """

    def __init__(self, output_dir: str, sahi_enabled: bool = False):
        super().__init__(output_dir, sahi_enabled)
        self.data_dir = os.path.join(output_dir, 'data')
        self.image_dir = os.path.join(self.data_dir, 'obj_train_data')
        os.makedirs(self.image_dir, exist_ok=True)

    def save_annotations(self, frame, frame_path: str, frame_filename: str, results, supported_classes: List[str]):
        """
        Saves annotations and frames in a format compatible with CVAT.
        """
        img_dimensions = frame.shape[:2]
        annotations = self.process_results(frame, results, img_dimensions)
        frame_filename_png = frame_filename.replace('.jpg', '.png')
        image_path = os.path.join(self.image_dir, frame_filename_png)
        cv2.imwrite(image_path, frame)
        self.write_annotations(frame_filename_png, annotations)
        self.create_metadata_files(supported_classes)

    def write_annotations(self, frame_filename: str, annotations: List[str]):
        """
        Writes annotations to a text file associated with each frame image.
        """
        annotation_filename = frame_filename.replace('.png', '.txt')
        annotation_path = os.path.join(self.image_dir, annotation_filename)
        try:
            with open(annotation_path, 'w') as file:
                for annotation in annotations:
                    file.write(annotation + "\n")
        except IOError as e:
            print(f"Error writing annotation file {annotation_path}: {str(e)}")

    def create_metadata_files(self, supported_classes: List[str]):
        """
        Creates necessary metadata files for a CVAT training setup, including class names and training configurations.
        """
        obj_names_path = os.path.join(self.data_dir, 'obj.names')
        obj_data_path = os.path.join(self.data_dir, 'obj.data')
        train_txt_path = os.path.join(self.data_dir, 'train.txt')

        try:
            with open(obj_names_path, 'w') as f:
                for cls in supported_classes:
                    f.write(f"{cls}\n")

            with open(obj_data_path, 'w') as f:
                f.write("classes = {}\n".format(len(supported_classes)))
                f.write("train = data/train.txt\n")
                f.write("names = data/obj.names\n")
                f.write("backup = backup/\n")

            with open(train_txt_path, 'w') as f:
                for image_file in os.listdir(self.image_dir):
                    if image_file.endswith('.png'):
                        f.write(f"data/obj_train_data/{image_file}\n")
        except IOError as e:
            print(f"Error writing metadata files: {str(e)}")

    def zip_and_cleanup(self):
        """
        Zips the processed data for transfer or storage and cleans up the directory structure.
        """
        zip_path = os.path.join(self.output_dir, 'cvat_data.zip')
        try:
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(self.data_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, self.data_dir))
                    for dir in dirs:
                        dir_path = os.path.join(root, dir)
                        zipf.write(dir_path, os.path.relpath(dir_path, self.data_dir))

            # Cleanup
            for root, dirs, files in os.walk(self.data_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            os.rmdir(self.data_dir)
        except Exception as e:
            print(f"Error during zip or cleanup: {str(e)}")

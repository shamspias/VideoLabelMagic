from typing import Optional, List, Dict


class BaseFormat:
    """
    Base class for handling annotation formats. Provides foundational functionalities
    like saving annotations and ensuring directory structure, designed for extension by subclasses.

    Attributes:
        output_dir (str): Directory where output will be stored.
        sahi_enabled (bool): Flag to enable or disable SAHI (Sliced Inference).
        sahi_utils (Optional[object]): SAHI utility object for performing sliced inference.
    """

    def __init__(self, output_dir: str, sahi_enabled: bool = False, sahi_utils: Optional[object] = None):
        """
        Initializes the BaseFormat class with output directory and optional SAHI settings.

        Args:
            output_dir (str): Path to the directory where annotations will be saved.
            sahi_enabled (bool): Boolean flag to enable SAHI (Sliced Inference). Defaults to False.
            sahi_utils (Optional[object]): Instance of SAHI utility class to be used for sliced inference. Defaults to None.
        """
        self.output_dir = output_dir
        self.sahi_enabled = sahi_enabled
        self.sahi_utils = sahi_utils

    def write_annotations(self, frame_filename: str, annotations: List[str]):
        """
        Writes annotations to a file based on the frame filename.

        Args:
            frame_filename (str): The filename of the frame to which annotations relate.
            annotations (List[str]): Annotations to be written to the file.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def ensure_directories(self):
        """
        Ensures that necessary directories for saving annotations exist.
        Must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def process_results(self, results: Dict, img_dimensions, supported_classes) -> List[str]:
        """
        Generate formatted strings from detection results suitable for annotations.

        Args:
            frame: The image frame being processed.
            results: Detection results containing bounding boxes and class IDs.
            img_dimensions: Dimensions of the image for normalizing coordinates.

        Returns:
            List of annotation strings formatted according to specific requirements.
        """
        annotations = []
        img_height, img_width = img_dimensions

        # Check if SAHI is enabled to adapt processing of results accordingly
        if self.sahi_enabled:
            for box in results['boxes']:  # Assuming SAHI results are formatted similarly
                class_id = int(box['cls'][0])
                xmin, ymin, xmax, ymax = box['xyxy'][0]
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height
                annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
        else:
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        xmin, ymin, xmax, ymax = box.xyxy[0]
                        x_center = ((xmin + xmax) / 2) / img_width
                        y_center = ((ymin + ymax) / 2) / img_height
                        width = (xmax - xmin) / img_width
                        height = (ymax - ymin) / img_height
                        annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return annotations

    def save_annotations(self, frame, frame_path: str, frame_filename: str, results: Dict,
                         supported_classes: List[str]):
        """
        Abstract method for saving annotations. To be implemented by subclasses to define
        the logic for saving the annotations.

        Args:
            frame (ndarray): The image frame for which annotations are being saved.
            frame_path (str): The path where the frame is located.
            frame_filename (str): The name of the frame file.
            results (Dict): A dictionary of results from the detection model or sliced inference.
            supported_classes (List[str]): List of supported class labels for the annotations.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

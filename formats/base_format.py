from typing import Optional


class BaseFormat:
    """
    Base class for handling annotation formats. This class provides basic functionalities
    like saving annotations and ensuring directory structure, which can be extended by subclasses.

    Attributes:
        output_dir (str): The directory where the output will be stored.
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

    def ensure_directories(self):
        """
        Ensures that the necessary directories for saving annotations exist.
        Must be implemented by subclasses.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def process_results(self, frame, results, img_dimensions):
        """Generate formatted strings from detection results."""
        annotations = []
        img_height, img_width = img_dimensions
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

    def save_annotations(self, frame, frame_path: str, frame_filename: str, results: list, supported_classes: list):
        """
        Saves the annotations for a given frame. If SAHI is enabled, performs sliced inference before saving.

        Args:
            frame (ndarray): The image frame for which annotations are being saved.
            frame_path (str): The path where the frame is located.
            frame_filename (str): The name of the frame file.
            results (list): A list of results from the detection model or sliced inference.
            supported_classes (list): List of supported class labels for the annotations.

        Raises:
            NotImplementedError: If `_save_annotations` is not implemented in the subclass.
        """
        self._save_annotations(frame, frame_path, frame_filename, results, supported_classes)

    def _save_annotations(self, frame, frame_path: str, frame_filename: str, results: list, supported_classes: list):
        """
        Abstract method for saving annotations. To be implemented by subclasses to define
        the logic for saving the annotations.

        Args:
            frame (ndarray): The image frame for which annotations are being saved.
            frame_path (str): The path where the frame is located.
            frame_filename (str): The name of the frame file.
            results (list): A list of results from the detection model or sliced inference.
            supported_classes (list): List of supported class labels for the annotations.

        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses should implement this method.")

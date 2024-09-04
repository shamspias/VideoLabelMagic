class BaseFormat:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.image_dir = None
        self.label_dir = None

    def ensure_directories(self):
        """
        Ensures that necessary directories are created.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def annotate_frame(self, frame, frame_path, frame_filename, model_conf, supported_classes):
        """
        Annotates the frame using the model output and saves the annotation in a format-specific manner.

        Parameters:
            frame (np.array): The frame to be annotated.
            frame_path (str): Path where the frame image is saved.
            frame_filename (str): Filename of the frame image.
            model_conf (float): Model confidence threshold for annotations.
            supported_classes (list): List of supported class names.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def save_annotations(self, frame, frame_path, frame_filename, results, supported_classes):
        """ Method to save annotations; implemented in subclasses. """
        raise NotImplementedError("Subclasses should implement this method.")

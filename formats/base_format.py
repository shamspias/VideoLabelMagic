class BaseFormat:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def save_annotations(self, annotations):
        raise NotImplementedError("This method should be implemented by subclasses.")

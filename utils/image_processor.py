import cv2


class ImageProcessor:
    """
    A class for handling image transformations such as resizing, converting to grayscale, and adjusting RGB channels.

    Attributes:
        output_size (tuple): The desired output dimensions for resizing images (width, height).
    """

    def __init__(self, output_size=(640, 640)):
        """
        Initializes the ImageProcessor with a specified output size.

        Parameters:
            output_size (tuple): The width and height to which images should be resized, default is (640, 640).
        """
        self.output_size = output_size

    def resize_image(self, image):
        """
        Resizes an image to the specified output size.

        Parameters:
            image (np.array): The image to resize.

        Returns:
            np.array: The resized image.
        """
        resized_image = cv2.resize(image, self.output_size, interpolation=cv2.INTER_AREA)
        assert resized_image.shape[0] == self.output_size[1] and resized_image.shape[1] == self.output_size[
            0], "Resizing did not match expected dimensions."
        return resized_image

    def convert_to_grayscale(self, image):
        """
        Converts an image to grayscale.

        Parameters:
            image (np.array): The image to convert.

        Returns:
            np.array: The grayscale image.
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Input image is not in expected RGB format.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def rotate_image_90_degrees(self, image):
        """
        Rotates an image by 90 degrees clockwise.

        Parameters:
            image (np.array): The image to rotate.

        Returns:
            np.array: The rotated image.
        """
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

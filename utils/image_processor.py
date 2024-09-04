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
        return cv2.resize(image, self.output_size)

    def convert_to_grayscale(self, image):
        """
        Converts an image to grayscale.

        Parameters:
            image (np.array): The image to convert.

        Returns:
            np.array: The grayscale image.
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def adjust_rgb(self, image, red=1.0, green=1.0, blue=1.0):
        """
        Adjusts the RGB channels of an image.

        Parameters:
            image (np.array): The original image.
            red (float): The multiplier for the red channel.
            green (float): The multiplier for the green channel.
            blue (float): The multiplier for the blue channel.

        Returns:
            np.array: The RGB adjusted image.
        """
        b, g, r = cv2.split(image)
        adjusted_b = cv2.multiply(b, blue)
        adjusted_g = cv2.multiply(g, green)
        adjusted_r = cv2.multiply(r, red)
        return cv2.merge((adjusted_b, adjusted_g, adjusted_r))

import cv2
import numpy as np
from typing import Dict, List, Any


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Check that the image is in RGN format.

    In grayscale case convert the image to rhe RGB.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


class VisualDrawer:
    """Allows to visualize data.

    Args:
        config (Dict[str, Any]): Inference config.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _prediction_to_original_size(
            self, prediction: List[np.ndarray], height: int, width: int
    ) -> List[np.ndarray]:
        """Restore the mask to original image size.

        To process image by NN need to resize and in some cases pad
        image. So the mask have same coordinates as resize/padded image.

        Args:
            prediction (List[np.ndarray]): NN prediction.
            height (int): Image height.
            width (int): Image width.

        Returns:
            List[np.ndarray]: Restored prediction.
        """
        resize_value_h = 1. if self.config['image_size'][0] >= height \
            else height / self.config['image_size'][0]
        resize_value_w = 1. if self.config['image_size'][1] >= width \
            else width / self.config['image_size'][1]

        for idx in range(len(prediction)):
            prediction[idx] = np.float32(prediction[idx]) * \
                              np.array([resize_value_w, resize_value_h])
            prediction[idx] = np.int32(prediction[idx])
        return prediction

    def visualize_prediction(
            self, image: np.ndarray,
            prediction: List[np.ndarray]
    ) -> np.ndarray:
        """Draw contours on image.

        Args:
            image (np.ndarray(w, h, 3)): Image was loaded to NN.
            prediction (List[np.ndarray]): NN prediction.

        Returns:
            np.ndarray(w, h, 3): Image with contours.
        """
        image = to_rgb(image=image)
        height, width = image.shape[:2]

        prediction = self._prediction_to_original_size(prediction,
                                                       height, width)

        cv2.drawContours(image, prediction, -1, (0, 0, 255), 3)
        return image

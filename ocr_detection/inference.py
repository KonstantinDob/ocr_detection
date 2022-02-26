import cv2
import numpy as np
from typing import List

from ocr_detection.builder.builder import OCRDet
from ocr_detection.data.augmentations import to_tensor
from ocr_detection.visualizers.visualizer import to_rgb


class InferenceOCRDet(OCRDet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours on predicted mask.
        Args:
            mask (np.ndarray(w, h, nc)): Prediction mask.
        Returns:
            List[np.ndarray]: Contours on prediction mask.
        """
        output_contours = []

        mask = np.transpose(mask, (1, 2, 0))
        mask = np.uint8(
            np.where(mask > self.config['mask_threshold'], 255, 0)
        )

        mask = mask[:, :, 0] - mask[:, :, 1]

        # Threshold value is not important because there are only 0 and
        # 255 values in the mask.

        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.config['min_area']:
                output_contours.append(cnt[:, 0])

        return output_contours

    def visualize(self, image: np.ndarray,
                  prediction: List[np.ndarray]) -> np.ndarray:
        """Draw contours on image.
        Args:
            image (np.ndarray(w, h, 3)): Image was loaded to NN.
            prediction (List[np.ndarray]): NN prediction.
        Returns:
            np.ndarray: Image with contours.
        """
        self._visual.visualize_prediction(image=image,
                                          prediction=prediction)
        return image

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """The model make prediction on image.
        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).
        Args:
            image (np.ndarray(w, h, nc)): Input image. Can be RGB/BGR
                or grayscale.
        Returns:
            List[np.ndarray]: List with predicted contours.
        """
        image = to_rgb(image=image)
        image = self.augmentor.resize_normalize(image=image,
                                                is_mask=False)
        image = to_tensor(image).unsqueeze(0)
        prediction_mask = self.trainer.model.predict(
            image)[0].cpu().detach().numpy()

        prediction = self._find_contours(mask=prediction_mask)

        return prediction

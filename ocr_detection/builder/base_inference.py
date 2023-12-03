import abc
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

import torch.nn

from ocr_detection.data import Augmentor
from ocr_detection.visualizers import VisualDrawer
from ocr_detection.visualizers import LOGGER


class BaseInferenceOCRDet:
    """Inference OCR Detection model."""

    def __init__(self, config: Dict[str, Any]):
        """Base inference OCR detection constructor.

        Args:
            config (dist of str: Any): Config with init data.
        """
        LOGGER.info("Creating OCR Detection")
        self._augmentor: Optional[Augmentor] = None
        self.visual: Optional[VisualDrawer] = None
        self.model: Optional[torch.nn.Module] = None

        self.config = config
        self.device = self.config["device"]
        if self.config["mode"] == "inference":
            self.augmentor = Augmentor(False, self.config)
        self._create_modules()
        LOGGER.info("OCR Detection is created")

    def _find_contours(self, mask: np.ndarray) -> List[np.ndarray]:
        """Find contours on predicted mask.

        Args:
            mask (np.ndarray): Prediction mask.

        Returns:
            list of np.ndarray: Contours on prediction mask.
        """
        output_contours = []

        mask = np.transpose(mask, (1, 2, 0))
        mask = np.uint8(np.where(mask > self.config["mask_threshold"], 255, 0))

        mask = mask[:, :, 0] - mask[:, :, 1]

        # Threshold value is not important because there are only 0 and
        # 255 values in the mask.

        ret, thresh = cv2.threshold(mask, 127, 255, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > self.config["min_area"]:
                output_contours.append(cnt[:, 0])

        return output_contours

    def visualize(self, image: np.ndarray, prediction: List[np.ndarray]) -> np.ndarray:
        """Draw contours on image.

        Args:
            image (np.ndarray): Image was loaded to NN.
            prediction (list of np.ndarray): NN prediction.

        Returns:
            np.ndarray: Image with contours.
        """
        self.visual.visualize_prediction(image=image, prediction=prediction)
        return image

    @abc.abstractmethod
    def _create_modules(self) -> None:
        """Create modules."""
        pass

    @abc.abstractmethod
    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """Make prediction on the image.

        Args:
            image (np.ndarray): Image was loaded to NN.
        """
        pass

"""OCR detection infecence implementation."""

import numpy as np
from typing import List

from ocr_detection.model.model import create_model
from ocr_detection.visualizers.visualizer import VisualDrawer
from ocr_detection.data.augmentations import to_tensor
from ocr_detection.visualizers.visualizer import to_rgb
from ocr_detection.builder.base_inference import BaseInferenceOCRDet

from gyomei_trainer.model import Model


class InferenceOCRDet(BaseInferenceOCRDet):
    """Inference OCR Recognition model."""

    def __init__(self, *args, **kwargs):
        """OCR inference constructor.

        Args:
            args: Additional arguments for the parent class.
            kwargs: Additional keyword arguments for the parent class.
        """
        super().__init__(*args, **kwargs)

    def _create_modules(self) -> None:
        """Create Gyomei trainer."""
        model = create_model(self.config)

        self.visual = VisualDrawer(config=self.config)
        self.model = Model(model, None, None, self.config["device"])
        self.model.load_model(file_path=self.config["pretrained"])

    def predict(self, image: np.ndarray) -> List[np.ndarray]:
        """The model make prediction on image.

        Image should be a raw opencv-python type that is RGB/Grayscale
        np.uint8 (the pixel values in range 0-255).

        Args:
            image (np.ndarray): Input image. Can be RGB/BGR or grayscale.

        Returns:
            list of np.ndarray: List with predicted contours.
        """
        image = to_rgb(image=image)
        height, width = image.shape[:2]
        image = self.augmentor.resize_normalize(image=image, is_mask=False)
        image = to_tensor(image).unsqueeze(0)
        prediction_mask = self.model.predict(image)[0].cpu().detach().numpy()

        prediction = self._find_contours(mask=prediction_mask)
        prediction = self.visual.prediction_to_original_size(prediction, height, width)

        return prediction

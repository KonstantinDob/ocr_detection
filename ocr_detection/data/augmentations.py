import cv2
import numpy as np
from copy import deepcopy
from typing import Dict, Any, Optional

from torch import tensor, from_numpy
import albumentations as A

from ocr_detection.visualizers import LOGGER


def to_tensor(data: np.ndarray) -> tensor:
    """Data with shape [h, w, nc] which processed to pytorch tensor format.

    Args:
        data (np.ndarray): Raw data.

    Returns:
        tensor: Processed to pytorch tensor format data.
    """
    data = np.transpose(data, (2, 0, 1))
    data = from_numpy(data).float()
    return data


class Augmentor:
    """Class with basic augmenting functions."""

    def __init__(self, augment: bool, config: Dict[str, Any]):
        """Augmentor constructor.

        Args:
            augment (bool): Whether to use augmentation.
            config (dict of str: Any): Config with augmentation parameters.
                Can be None in case of turned off augmentations.
        """
        self.transform: Optional[A.Compose] = None

        self.config = config
        self.augment = augment
        self.config_aug = {}
        if self.augment:
            self.config_aug = config["augmentations"]

        self._create_albumentations()

    def _create_albumentations(self) -> None:
        """Create an albumentations transform."""
        if self.augment:
            image_comprehension = self.config_aug["image_comprehension"]
            self.transform = A.Compose(
                [
                    A.Blur(p=self.config_aug["blur"]),
                    A.MedianBlur(p=self.config_aug["median_blur"]),
                    A.ToGray(p=self.config_aug["gray"]),
                    A.CLAHE(p=self.config_aug["clahe"]),
                    A.RandomBrightnessContrast(p=self.config_aug["brightness_contrast"]),
                    A.RandomGamma(p=self.config_aug["gamma"]),
                    A.ImageCompression(
                        quality_lower=image_comprehension["quality_lower"],
                        p=image_comprehension["p"]
                    )
                ],
            )
            LOGGER.info("".join(f"Augmentation {item[0]}: {item[1]} \n"
                                for item in self.config_aug.items()))
        else:
            LOGGER.info("Augmentations turn off")

    def _normalize(self, image: np.ndarray, is_mask: bool) -> np.ndarray:
        """Normalize image.

        If mask is processed this one is divided by 255.

        Args:
            image (np.ndarray): Image.
            is_mask (bool): If image is RGB image or mask with contours.

        Returns:
            np.ndarray: Normalized data.
        """
        image = np.float32(image)
        if is_mask:
            image /= 255
        else:
            image = (image - self.config["mean"]) / self.config["std"]
        return image

    def _add_padding(self, image: np.ndarray) -> np.ndarray:
        """Add padding to image.

        If the size of the image on one of the sides is less than
        the specified one, then 0 padding will be added.

        Args:
            image (np.ndarray): Image.

        Returns:
            np.ndarray: Image with padding.
        """
        height, width, num_channel = image.shape
        req_height, req_width = self.config["image_size"]

        pad_image = deepcopy(image)
        if height < req_height or width < req_width:
            pad_image = np.zeros([max(height, req_height),
                                  max(width, req_width), num_channel])
            pad_image[:height, :width, :] = image

        return pad_image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize images.

        Args:
            image (np.ndarray): Image.

        Returns:
            np.ndarray: Reswized image.
        """
        image = cv2.resize(image, tuple(self.config["image_size"]))
        return image

    def albumentations(self, image: np.ndarray) -> np.ndarray:
        """Add albumentations augmentations to image.

        Args:
            image (np.ndarray): Raw image.

        Returns:
            np.ndarray: Processed image.
        """
        transformed = self.transform(image=image)
        transformed_image = transformed["image"]
        return transformed_image

    def rotation_aug(self, image: np.ndarray, rotate_vals: np.ndarray) -> np.ndarray:
        """Rotate image.

        Args:
            image (np.ndarray): Raw image.
            rotate_vals (np.array): Random probabilities to
                rotate image in upper-down and left-right directions.

        Returns:
            np.ndarray: Processed image.
        """
        if rotate_vals[0] < self.config_aug["flipud"]:
            image = np.flipud(image)
        if rotate_vals[1] < self.config_aug["fliplr"]:
            image = np.fliplr(image)

        return image

    def resize_normalize(self, image: np.ndarray, is_mask: bool) -> np.ndarray:
        """Resize and normalize image.

        Args:
            image (np.ndarray): Image.
            is_mask (bool): If image is RGB image or mask with contours.

        Returns:
            np.ndarray: Resized and normalized image.
        """
        image = self._normalize(image, is_mask=is_mask)
        image = self._add_padding(image)
        image = self._resize(image)
        return image

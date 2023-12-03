import os
from os.path import join

import cv2
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset

from ocr_detection.visualizers import LOGGER
from ocr_detection.visualizers import to_rgb
from ocr_detection.data import Augmentor
from ocr_detection.data import to_tensor


class OCRDetDataset(Dataset):
    """Load dataset for text detection tasks."""

    def __init__(self, mode: str, config: Optional[Dict[str, Any]] = None):
        """OCR dataset constructor.

        Args:
            mode (str): What part of the dataset should be loaded.
                Can take the following values {'train', 'test', 'tal'}.
            config (Optional[Dict[str, Any]]): Dataset config. If None
                load config from default path. Default is None.

        Raises:
            KeyError: Raises when incorected mode passed.
        """
        if mode not in ["train", "test", "val"]:
            raise KeyError("Mode entered incorrectly!")

        LOGGER.info(f"Creating {mode} OCRDetDataset")

        self.mode = mode
        self.config = config

        self.augment = mode == "train"
        self.augmentor = Augmentor(self.augment, self.config)

        self.images = []
        self.labels = []
        self.load_data()
        LOGGER.info("OCRDetDataset created")

    def load_data(self) -> None:
        """Load dataset in current mode.

        Raises:
            FileNotFoundError: Raise when datast is empty.
        """
        if self.config["is_dummy"] is True:
            self.images, self.labels = self.generate_dummy_data(num_samples=10)
            LOGGER.info("Dummy dataset is created")
            return

        path = self.config["datapath"]
        labels_path = join(path, f"{self.mode}_label")
        images_path = join(path, "images")

        labels_name = os.listdir(labels_path)
        images_name = [label_name[:-3] + "jpg" for label_name in labels_name]
        labels_name = [join(labels_path, lbl_name) for lbl_name in labels_name]
        images_name = [join(images_path, img_name) for img_name in images_name]

        LOGGER.info(f"Loading data. In memory: {self.config['all_in_memory']}")
        if self.config["all_in_memory"]:
            pbar = tqdm(zip(images_name, labels_name))
            for image_name, label_name in pbar:
                pbar.set_description(f"Processing {len(labels_name)} pairs")
                image = cv2.imread(image_name)
                image = to_rgb(image=image)

                self.images.append(np.uint8(image))
                self.labels.append(np.uint8(np.load(label_name)))
        else:
            self.images = images_name
            self.labels = labels_name

        if [] in [self.images, self.labels]:
            raise FileNotFoundError("Dataset is empty!")

        LOGGER.info("Data have loaded")

    def get_pair(self, index) -> Tuple[np.ndarray, np.ndarray]:
        """Get image/label pair. Label processed to mask.

        Args:
            index (int):The index of the returned image/label pair in
                the dataset.

        Returns:
            tuple of nd.array: Image/mask pair in the dataset.
        """
        if self.config["all_in_memory"]:
            return self.images[index], self.labels[index]
        else:
            image = np.uint8(cv2.imread(self.images[index]))
            label = np.uint8(np.load(self.labels[index]))
            return image, label

    def generate_dummy_data(self, num_samples: int = 15) -> List[np.ndarray]:
        """Generate dummy data.

        Args:
            num_samples: How much data should be generated.

        Returns:
            list of np.ndarray: generated labels and images.
        """
        image_size = self.config["image_size"] + [3]
        images = [
            np.uint8(np.random.randint(0, 255, size=image_size)) for _ in range(num_samples)
        ]
        labels = [np.uint8(np.zeros(image_size[:2])) for _ in range(num_samples)]
        return images, labels

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            int: Length of the dtaset.
        """
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get image/mask pair for training and process it to NN.

        Args:
            index (int):The index of the returned image/mask pair in the dataset.

        Returns:
            Tuple[np.ndarray(w, h, 3), np.ndarray(w, h, nc)]: Image/mask
                pair in the dataset prepared for loading into NN.
        """
        image, mask = self.get_pair(index)
        if self.augment:
            image = self.augmentor.albumentations(image)
            rotate_vals = np.random.random(2)
            image = self.augmentor.rotation_aug(image, rotate_vals=rotate_vals)
            mask = self.augmentor.rotation_aug(mask, rotate_vals=rotate_vals)

        image = self.augmentor.resize_normalize(image, is_mask=False)
        mask = self.augmentor.resize_normalize(mask, is_mask=True)
        image = to_tensor(image)
        mask = to_tensor(mask)
        return image, mask


def create_dataloaders_pair(main_config: Dict[str, Any]):
    """Create dataloaders for eval/train mode.

    Args:
        main_config (Dict[str, Any]): Config with initial data.

    Returns:
        tuple of DataLoader: Tuple with train and valid dataloader.

    Raises:
        KeyError: Raises when incorected mode passed.
    """
    mode = main_config["mode"]

    if mode == "eval":
        test_dataloader = create_dataloader(1, mode="test", config=main_config["data"])
        train_dataloader = None
    elif mode == "inference":
        test_dataloader = None
        train_dataloader = None
    elif mode == "train":
        batch_size = main_config["batch_size"]
        test_dataloader = create_dataloader(1, mode="val", config=main_config["data"])
        train_dataloader = create_dataloader(batch_size, mode="train", config=main_config["data"])
    else:
        raise KeyError("Incorrect mode!")

    return train_dataloader, test_dataloader


def create_dataloader(batch_size: int, mode: str, config: Dict[str, Any]) -> DataLoader:
    """Create dataloader with required properties.

    Args:
        batch_size (int): Required batch size.
        mode (str): What mode should be loaded.
        config (Dict[str, Any]): Dataset config. Path is: project/configs/data/dataset.yaml.

    Returns:
        DataLoader: Created dataloader.
    """
    LOGGER.info("Create Dataloader")
    dataset = OCRDetDataset(mode, config)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    LOGGER.info(f"Dataloader created with batch_size: {batch_size}")
    return dataloader

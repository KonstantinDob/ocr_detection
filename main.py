import torch
from torch.utils.data import DataLoader, Dataset

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses

from gyomei_trainer.model import Model
from gyomei_trainer.builder import Builder

import cv2
import yaml
import os
from tqdm import tqdm
from os.path import join
import numpy as np
from copy import deepcopy
import logging

from torch import tensor, from_numpy
import albumentations as A


def to_rgb(image: np.ndarray) -> np.ndarray:
    """Check that the image is in RGN format.

    In grayscale case convert the image to rhe RGB.
    """
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def to_tensor(data: np.ndarray) -> tensor:
    """Data with shape [h, w, nc] which processed to pytorch
    tensor format.

    Args:
        data (np.ndarray(w, h, nc)): Raw data.

    Returns:
        tensor: Processed to pytorch tensor format data.
    """
    data = np.transpose(data, (2, 0, 1))
    data = from_numpy(data).float()
    return data


class Augmentor:
    """Class with basic augmenting functions.

    Args:
        augment (bool): Whether to use augmentation.
        config (Dict[str, Any]): Config with augmentation
            parameters. Can be None in case of turned off augmentations.
    """

    def __init__(self, augment, config):
        self.config = config
        self.config_aug = config['augmentations']

        self.augment = augment
        self.transform = None
        self._create_albumentations()

    def _create_albumentations(self):
        """Create an albumentations transform."""
        if self.augment:
            image_comprehension = self.config_aug['image_comprehension']
            self.transform = A.Compose([
                A.Blur(p=self.config_aug['blur']),
                A.MedianBlur(p=self.config_aug['median_blur']),
                A.ToGray(p=self.config_aug['gray']),
                A.CLAHE(p=self.config_aug['clahe']),
                A.RandomBrightnessContrast(p=self.config_aug[
                    'brightness_contrast']),
                A.RandomGamma(p=self.config_aug['gamma']),
                A.ImageCompression(
                    quality_lower=image_comprehension['quality_lower'],
                    p=image_comprehension['p'])],
            )
        else:
            pass

    def _normalize(self, image: np.ndarray,
                   is_mask: bool) -> np.ndarray:
        """Normalize image.

        If mask is processed this one is divided by 255.
        """
        image = np.float32(image)
        if is_mask:
            image /= 255
        else:
            image = (image - self.config['mean']) / self.config['std']
        return image

    def _add_padding(self, image: np.ndarray) -> \
            np.ndarray:
        """Add padding to image.

        If the size of the image on one of the sides is less than
        the specified one, then 0 padding will be added.
        """
        height, width, num_channel = image.shape
        req_height, req_width = self.config['image_size']

        pad_image = deepcopy(image)
        if height < req_height or width < req_width:
            pad_image = np.zeros([max(height, req_height),
                                  max(width,  req_width), num_channel])
            pad_image[:height, :width, :] = image

        return pad_image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize images."""
        image = cv2.resize(image, tuple(self.config['image_size']))
        return image

    def albumentations(self, image: np.ndarray) -> np.ndarray:
        """Add albumentations augmentations to image.

        Args:
            image (np.ndarray(w, h, 3)): Raw image.

        Returns:
            np.ndarray: Processed image.
        """
        transformed = self.transform(image=image)
        transformed_image = transformed['image']
        return transformed_image

    def rotation_aug(self, image: np.ndarray,
                     rotate_vals: np.ndarray) -> np.ndarray:
        """Rotate image.

        Args:
            image (np.ndarray(w, h, 3)): Raw image.
            rotate_vals (np.array of 2): Random probabilities to
                rotate image in upper-down and left-right directions.

        Returns:
            np.ndarray: Processed image.
        """
        if rotate_vals[0] < self.config_aug['flipud']:
            image = np.flipud(image)
        if rotate_vals[1] < self.config_aug['fliplr']:
            image = np.fliplr(image)

        return image

    def resize_normalize(self, image: np.ndarray, is_mask: bool) -> \
            np.ndarray:
        """Resize and normalize image."""
        image = self._normalize(image, is_mask=is_mask)
        image = self._add_padding(image)
        image = self._resize(image)
        return image


class GyomeiDetDataset(Dataset):
    """Load dataset for text detection tasks.

    Args:
        mode (str): What part of the dataset should be loaded.
            Can take the following values {'train', 'test', 'tal'}.
        config (Optional[Dict[str, Any]]): Dataset config. If None
            load config from default path. Default is None.
    """

    def __init__(self, mode: str,
                 config=None):
        if mode not in ['train', 'test', 'val']:
            raise KeyError("Mode entered incorrectly!")

        self.mode = mode
        if config is None:
            with open('./configs/data/dataset.yaml', 'r') as file:
                self.config = yaml.safe_load(file)
        else:
            self.config = config

        self.augment = mode == 'train'
        self.augmentor = Augmentor(self.augment, self.config)

        self.images = []
        self.labels = []
        self.load_data()

    def load_data(self):
        """Load dataset in current mode."""
        path = self.config['datapath']
        labels_path = join(path, f'{self.mode}_label')
        images_path = join(path, 'images')

        labels_name = os.listdir(labels_path)
        images_name = [label_name[:-3] + 'jpg'
                       for label_name in labels_name]
        labels_name = [join(labels_path, lbl_name) for
                       lbl_name in labels_name]
        images_name = [join(images_path, img_name) for
                       img_name in images_name]

        if self.config['all_in_memory']:
            pbar = tqdm(zip(images_name, labels_name))
            for image_name, label_name in pbar:
                pbar.set_description(
                    f'Processing {len(labels_name)} pairs')
                image = cv2.imread(image_name)
                image = to_rgb(image=image)

                self.images.append(np.uint8(image))
                self.labels.append(np.uint8(np.load(label_name)))
        else:
            self.images = images_name
            self.labels = labels_name

        if [] in [self.images, self.labels]:
            raise FileNotFoundError("Dataset is empty!")

    def get_pair(self, index):
        """Get image/label pair. Label processed to mask.

        Args:
            index (int):The index of the returned image/label pair in
                the dataset.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Image/mask pair in
                the dataset.
        """
        if self.config['all_in_memory']:
            return self.images[index], self.labels[index]
        else:
            image = cv2.imread(np.uint8(self.images[index]))
            label = np.load(np.uint8(self.labels[index]))
            return image, label

    def __len__(self):
        """Get length of dataset."""
        return len(self.labels)

    def __getitem__(self, index: int):
        """Get image/mask pair for training and process it to NN.

        Args:
            index (int):The index of the returned image/mask pair in
                the dataset.

        Returns:
            Tuple[np.ndarray(w, h, 3), np.ndarray(w, h, nc)]: Image/mask
                pair in the dataset prepared for loading into NN.
        """
        image, mask = self.get_pair(index)
        if self.augment:
            image = self.augmentor.albumentations(image)
            rotate_vals = np.random.random(2)
            image = self.augmentor.rotation_aug(image,
                                                rotate_vals=rotate_vals)
            mask = self.augmentor.rotation_aug(mask,
                                               rotate_vals=rotate_vals)

        image = self.augmentor.resize_normalize(image, is_mask=False)
        mask = self.augmentor.resize_normalize(mask, is_mask=True)
        image = to_tensor(image)
        mask = to_tensor(mask)
        return image, mask


def create_dataloader(mode: str) -> DataLoader:
    with open('./configs/train.yaml', 'r') as file:
        config = yaml.safe_load(file)
    dataset = GyomeiDetDataset(mode)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config['batch_size'],
                            num_workers=config['num_workers'],
                            shuffle=True)
    return dataloader


def main():
    train_dataloader = create_dataloader(mode='train')
    valid_dataloader = create_dataloader(mode='val')

    smp_model = smp.Unet(
        encoder_name='resnet34',
        encoder_weights='imagenet',
        in_channels=3,
        classes=2
    )
    optimizer = torch.optim.Adam(
        params=smp_model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.999)

    metrics = dict()
    metrics['fscore'] = smp.utils.metrics.Fscore(threshold=0.5)
    metrics['iou'] = smp.utils.metrics.IoU(threshold=0.5)

    main_metric = ['iou']

    loss = losses.JaccardLoss(mode='multilabel', smooth=0)

    model = Model(model=smp_model, optimizer=optimizer,
                  loss=loss, device='cuda')
    # model.load_model('./experiments/Unet_resnet34_0'
    #                  '/models/best_Unet.pth')
    trainer = Builder(model=model, train_loader=train_dataloader,
                      valid_loader=valid_dataloader, num_epoch=20,
                      metrics=metrics, main_metrics=main_metric,
                      scheduler=scheduler, early_stopping_patience=5,
                      seed=666)

    trainer.fit()


if __name__ == "__main__":
    main()

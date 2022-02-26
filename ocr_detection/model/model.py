import torch
from typing import Dict, Any
from segmentation_models_pytorch import Unet
from ocr_detection.visualizers.logger import LOGGER


def create_model(main_config: Dict[str, Any]) -> torch.nn.Module:
    """Create model based on Segmentation Models Pytorch (SMP).
    More info about SMP:
    https://github.com/qubvel/segmentation_models.pytorch .

    Args:
        main_config (Dict[str, Any]): Config with initial data.

    Returns:
        torch.nn.Module: Created model.
    """
    config = main_config['model']

    if config['base'] == 'Unet':
        model = Unet(
            encoder_name=config['encoder_name'],
            encoder_weights=config['encoder_weights'],
            in_channels=config['in_channels'],
            classes=config['classes']
        )
    else:
        raise KeyError('Incorrect model name!')

    LOGGER.info(
        f"{config['base']} model is created with following "
        f"properties:\n"
        f"Encoder: {config['encoder_name']},\n"
        f"Input channels: {config['in_channels']},\n"
        f"Num of classes: {config['classes']}."
    )
    return model

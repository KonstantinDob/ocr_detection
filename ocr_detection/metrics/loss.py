import torch
from typing import Dict, Optional, Any

from segmentation_models_pytorch import losses

from ocr_detection.visualizers import LOGGER


def create_loss_function(main_config: Dict[str, Any]) -> Optional[torch.nn.Module]:
    """Create loss function also based on SMP.

    Args:
        main_config (dict os str: Any): Config with initial data.

    Returns:
        torch.nn.Module, optional: Created loss function.

    Raises:
        KeyError: Raise when incorrect loss name pass.
    """
    if "loss_function" not in main_config:
        return

    config = main_config["loss_function"]

    if config["name"] == "JaccardLoss":
        loss = losses.JaccardLoss(mode="multilabel", smooth=config["smooth"])
    elif config["name"] == "DiceLoss":
        loss = losses.DiceLoss(mode="multilabel", smooth=config["smooth"])
    else:
        raise KeyError("Incorrect loss name!")

    LOGGER.info(
        f"{config['name']} is created with following properties: Smooth: {config['smooth']}."
    )
    return loss

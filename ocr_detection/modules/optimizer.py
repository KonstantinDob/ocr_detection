import torch
from typing import Dict, Optional, Any

from ocr_detection.visualizers import LOGGER


def create_optimizer(
    main_config: Dict[str, Any], model: torch.nn.Module
) -> Optional[torch.optim.Optimizer]:
    """Create optimizer.

    Args:
        main_config (dict of str: Any): Config with initial data.
        model (torch.nn.Module): That model should be trained.

    Returns:
        torch.optim.Optimizer, optional: Created optimizer.

    Raises:
        KeyError: Raise when incorrect optimizer name pass.
    """
    if "optimizer" not in main_config:
        return

    config = main_config["optimizer"]
    if config["name"] == "Adam":
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=config["lr"],
            betas=(config["momentum"], 0.999)
        )
    elif config["name"] == "SGD":
        optimizer = torch.optim.SGD(
            params=model.parameters(),
            lr=config["lr"],
            momentum=config["momentum"]
        )
    else:
        raise KeyError("Incorrect optimizer name!")

    LOGGER.info(
        f"{config['name']} optimizer is created with following "
        f"properties:\n"
        f"Learning rate: {config['lr']},\n"
        f"Momentum: {config['momentum']}."
    )
    return optimizer

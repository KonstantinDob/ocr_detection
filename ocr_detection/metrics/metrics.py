from typing import Dict, Optional, Any

from segmentation_models_pytorch.utils import metrics

from ocr_detection.visualizers import LOGGER


def create_metrics(main_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Create metrics based on SMP.

    Args:
        main_config (dict of str: Any): Config with initial data.

    Returns:
        dict of str: Any, optional: Dict with metrics.

    Raises:
        AttributeError: Raise when metrics not pass.
        KeyError: Raise when incorrect metric name pass.
    """
    if "metrics" not in main_config:
        return

    metrics_data = main_config["metrics"]
    metrics_dict = {}

    if metrics_data["types"] == []:
        raise AttributeError("Metrics must be not empty list.")

    for metric in metrics_data["types"]:
        if metric == "f_score":
            metrics_dict[metric] = metrics.Fscore(threshold=metrics_data["threshold"])
        elif metric == "precision":
            metrics_dict[metric] = metrics.Precision(threshold=metrics_data["threshold"])
        elif metric == "recall":
            metrics_dict[metric] = metrics.Recall(threshold=metrics_data["threshold"])
        elif metric == "iou_score":
            metrics_dict[metric] = metrics.IoU(threshold=metrics_data["threshold"])
        else:
            raise KeyError("Incorrect metric name!")

        LOGGER.info(f"{metric} is used.")
    return metrics_dict

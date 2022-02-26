from typing import Dict, Optional, Any
from segmentation_models_pytorch.utils import metrics
from ocr_detection.visualizers.logger import LOGGER


def create_metrics(main_config: Dict[str, Any]) -> \
        Optional[Dict[str, Any]]:
    """Create metrics based on SMP.

    Args:
        main_config (Dict[str, Any]): Config with initial data.

    Returns:
        Optional[Dict[str, Any]]: Dict with metrics.
    """
    if 'metrics' not in main_config:
        return

    metrics_data = main_config['metrics']
    metrics_dict = {}

    assert metrics_data['types'] != [], \
        "Metrics must be not empty list."

    for metric in metrics_data['types']:
        if metric == 'f_score':
            metrics_dict[metric] = metrics.Fscore(
                threshold=metrics_data['threshold'])
        elif metric == 'precision':
            metrics_dict[metric] = metrics.Precision(
                threshold=metrics_data['threshold'])
        elif metric == 'recall':
            metrics_dict[metric] = metrics.Recall(
                threshold=metrics_data['threshold'])
        elif metric == 'iou_score':
            metrics_dict[metric] = metrics.IoU(
                threshold=metrics_data['threshold'])
        else:
            raise KeyError('Incorrect metric name!')

        LOGGER.info(f"{metric} is used.")
    return metrics_dict

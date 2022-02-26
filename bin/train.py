import yaml
from typing import Dict, Any
from ocr_detection.builder.builder import OCRDet


def create_main_config() -> Dict[str, Any]:
    """Merge all configs to one.

    Returns:
        Dict[str, Any]: Main config that contain all data.
    """
    with open('/home/konstantin/workdir/ocr_detection'
              '/configs/data/dataset.yaml', 'r') as file:
        data_config = yaml.safe_load(file)
    with open('/home/konstantin/workdir/ocr_detection'
              '/configs/model/model.yaml', 'r') as file:
        model_config = yaml.safe_load(file)
    with open('/home/konstantin/workdir/ocr_detection/'
              'configs/train.yaml', 'r') as file:
        train_config = yaml.safe_load(file)

    config = {}
    config.update(train_config)
    config['model'] = model_config
    config['data'] = data_config
    return config


def main():
    config = create_main_config()
    ocr_det = OCRDet(config)
    ocr_det.train()


if __name__ == "__main__":
    main()

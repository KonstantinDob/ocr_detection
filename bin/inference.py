import os
import cv2
import yaml
import argparse
from tqdm import tqdm
from typing import Dict, Any

from ocr_detection.inference import InferenceOCRDet


def create_main_config() -> Dict[str, Any]:
    """Merge all configs to one.

    Returns:
        Dict[str, Any]: Main config that contain all data.
    """
    with open('./configs/inference.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--datapath',
        type=str,
        default='./dataset/inference'
    )

    args = parser.parse_args()
    return args


def main():
    """Before run inference set parameters in the inference.yaml."""
    config = create_main_config()

    inference = InferenceOCRDet(config)
    args = parse_arguments()

    image_names = os.listdir(args.datapath)
    image_paths = [os.path.join(args.datapath, image_name) for image_name in image_names]
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        prediction = inference.predict(image=image)
        image = inference.visualize(image=image, prediction=prediction)

        cv2.imshow('Result', image)
        cv2.waitKey()


if __name__ == "__main__":
    main()

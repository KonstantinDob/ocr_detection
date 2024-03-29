import yaml
import pytest
from typing import Dict, Any

from ocr_detection.inference import InferenceOCRDet


@pytest.fixture
def create_main_config() -> Dict[str, Any]:
    """Merge all configs to one.

    Returns:
        Dict[str, Any]: Main config that contain all data.
    """
    with open('./configs/inference.yaml', 'r') as file:
        config = yaml.safe_load(file)

    return config


class TestInference:

    def test_init(self, create_main_config):
        """Test inference initialization.

        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!
        """
        import os

        print(os.path.exists('./configs/inference.yaml'))
        print(os.path.abspath(os.getcwd()))

        if not os.path.exists("./dataset"):
            return

        config = create_main_config

        inference = InferenceOCRDet(config)

        state = inference.trainer.state

        assert inference.augmentor is not None
        assert inference.trainer.valid_loader is None
        assert inference.trainer.train_loader is None

        assert state.epoch == 0
        assert state.folder_path is None
        assert state.main_metrics is None

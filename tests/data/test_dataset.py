import yaml
import pytest
from typing import Dict, Any

from ocr_detection.data.datasets import OCRDetDataset


@pytest.fixture
def load_config() -> Dict[str, Any]:
    """Load dataset config.

    Returns:
        Dict[str, Any]: Dataset config.

    """
    with open('./configs/data/dataset.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config


class TestDataset:

    @pytest.mark.parametrize(
        'datapath, created', [('same', True), ('', False),
                              ('./empty_dataset', False)]
    )
    def test_dataset_load(self, load_config,
                          datapath: str, created: bool):
        """Test creating dataset.
        Run in case of [empty, not empty, incorrect path for] dataset.

        Args:
            datapath (str): path to dataset.
            created (str): Should dataset was created.

        """
        config = load_config
        # Use it when you can not get the data
        config["is_dummy"] = False if created is False else True

        if datapath != 'same':
            config['datapath'] = datapath

        for mode in ['train', 'test', 'val']:
            try:
                OCRDetDataset(mode=mode, config=config)
                # dataset created properly
                assert created
            except FileNotFoundError:
                # failed to create dataset
                assert not created

    @pytest.mark.parametrize(
        'mode, created', [('val', True), ('', False), ('lal', False),
                          ('train', True), ('test', True)]
    )
    def test_dataset_modes(self, load_config, mode: str, created):
        """Create dataset with various modes.
        Before the test CHECK your dataset.yaml. In this config
        datapath parameter should be correct path to the dataset!

        Args:
            mode (str): In which mode should create dataset.
            created (bool):  Should dataset was created.

        """
        config = load_config

        try:
            OCRDetDataset(mode=mode, config=config)
            # dataset created properly
            assert created
        except KeyError:
            # failed to create dataset
            assert not created

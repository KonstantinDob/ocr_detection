from typing import Dict, Any, Optional

from ocr_detection.model import create_model
from ocr_detection.data import Augmentor
from ocr_detection.data import create_dataloaders_pair
from ocr_detection.metrics import create_metrics, create_loss_function
from ocr_detection.modules import create_optimizer, create_scheduler
from ocr_detection.visualizers import LOGGER
from ocr_detection.visualizers import VisualDrawer

from gyomei_trainer.model import Model
from gyomei_trainer.builder import Builder


class OCRDet:
    """Class for training and evaluation."""

    def __init__(self, config: Dict[str, Any]):
        """OCR detection class constructor.

        Args:
            config (Dict[str, Any]): Config with initial data.
        """
        LOGGER.info("Creating OCR Detection")
        self.augmentor: Optional[Augmentor] = None
        self._visual: Optional[VisualDrawer] = None
        self.trainer: Optional[Builder] = None

        self.config = config
        self.device = self.config["device"]
        if self.config["mode"] == "inference":
            self.augmentor = Augmentor(False, self.config)
            self._visual = VisualDrawer(config=self.config)
        self._create_modules()
        LOGGER.info("OCR Detection is created")

    def _create_modules(self):
        """Create Gyomei trainer."""
        model = create_model(self.config)
        optimizer = create_optimizer(self.config, model)
        loss = create_loss_function(self.config)
        metrics_dict = create_metrics(self.config)
        scheduler = create_scheduler(self.config, optimizer)
        train_dataloader, test_dataloader = \
            create_dataloaders_pair(self.config)

        data = self._get_aux_params()

        gyomei_model = Model(model, optimizer,
                             loss, self.config["device"])
        self.trainer = Builder(
            model=gyomei_model, train_loader=train_dataloader,
            valid_loader=test_dataloader, num_epoch=data["epoch"],
            metrics=metrics_dict, main_metrics=data["main_metrics"],
            scheduler=scheduler, project_path=data["project_path"],
            early_stopping_patience=data["patience"]
        )

    def _get_aux_params(self):
        """Get few auxiliary parameters from config.

        Returns:
            Dict[str, Any]: Dict with auxiliary parameters.
        """
        data = {"epoch": None, "main_metrics": None, "patience": None, "project_path": None}
        for key, val in data.items():
            if key in self.config:
                data[key] = self.config[key]
        return data

    def train(self):
        """Run gyomei training session."""
        self.trainer.fit()

    def eval(self):
        """Run gyomei evaluation session."""
        self.trainer.valid_epoch()

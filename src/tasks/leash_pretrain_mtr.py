from typing import Any
from joblib import Parallel, delayed
from .base import BaseTask
import logging
import hydra
from torch.nn import functional as F
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from data.datasets.base import BaseDataset
import time

logger = logging.getLogger(__name__)


class LeashPretrainMTRTask(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)

        model_cls = hydra.utils.get_class(cfg.model._target_)
        self.model = model_cls(cfg)
        if getattr(cfg.env, "log_model", True):
            logger.info("--------MODEL--------\n%s", self.model)

        # self.train_epoch_idxs = []

    def forward(self, input_ids, padding_mask):
        return self.model(input_ids, padding_mask)

    def _shared_step(self,
                     model,
                     stage,
                     batch,
                     batch_idx,
                     dataloader_idx=None):
        mtr_target = batch["mtr_target"]
        pred = model(batch['input_ids'], batch['padding_mask'])
        loss = self.criterion(pred, mtr_target)
        step_metrics = {
            f"{stage}/loss": loss,
        }
        step_output = {"loss": loss, "pred": pred, "target": mtr_target}
        return step_output, step_metrics

    def _eval_step_single_model(self,
                                model,
                                batch,
                                batch_idx,
                                dataloader_idx=None,
                                stage="val"):
        step_output, step_metrics = self._shared_step(
            model, stage, batch, batch_idx, dataloader_idx=dataloader_idx)
        step_save_output = {}
        return step_output, step_save_output, step_metrics

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        # self.train_epoch_idxs.extend(batch["idx"].cpu().numpy())
        return super().training_step(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        raise NotImplementedError
        pred = self(batch['input_ids'], batch['padding_mask'])
        pred = F.sigmoid(pred).float().cpu().numpy()
        self.predict_step_outputs.append(pred)
        return pred

    def on_train_epoch_end(self):
        # logger.info("UNIQUE TRAIN IDXS: %d", len(set(self.train_epoch_idxs)))
        # self.train_epoch_idxs.clear()
        return super().on_train_epoch_end()

    # @override
    def _compute_metrics(self,
                         step_outputs: dict,
                         dataset: BaseDataset,
                         stage: str = "val"):
        raise NotImplementedError

    # @override
    def _on_epoch_end_save_metadata(self, metadata, stage, model_name):
        return super()._on_epoch_end_save_metadata(metadata, stage, model_name)

    def on_validation_epoch_end(self):
        return super().on_validation_epoch_end()

    def on_test_epoch_end(self) -> None:
        # post processing
        # calculate & log metric
        return super().on_test_epoch_end()

    def on_predict_epoch_end(self):
        return super().on_predict_epoch_end()

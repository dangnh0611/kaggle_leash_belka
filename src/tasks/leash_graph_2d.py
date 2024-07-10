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
from src.tasks.leash import LeashTask

logger = logging.getLogger(__name__)


class LeashGraph2DTask(LeashTask):

    def forward(self, batch):
        return self.model(batch)
    
    def get_batch_size(self, batch):
        return batch.batch.max()

    def _shared_step(self,
                     model,
                     stage,
                     batch,
                     batch_idx,
                     dataloader_idx=None):
        target = batch.target
        pred = model(batch)
        loss = self.criterion(pred, target)
        step_metrics = {
            f"{stage}/loss": loss,
        }
        step_output = {"loss": loss, "pred": pred, "target": target}
        return step_output, step_metrics

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pred = self(batch)
        if self.task_type == 'cls':
            pred = F.sigmoid(pred)
        pred = pred.float().cpu().numpy()
        self.predict_step_outputs.append(pred)
        return pred

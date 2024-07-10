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
import torch

logger = logging.getLogger(__name__)


class LeashTask(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)

        model_cls = hydra.utils.get_class(cfg.model._target_)
        model = model_cls(cfg)
        self.model = model

        if getattr(cfg.env, "log_model", True):
            logger.info("--------MODEL--------\n%s", self.model)

        # default to classification task with sigmoid
        self.task_type = getattr(cfg.task, 'type', 'cls')
        # self.train_epoch_idxs = []

    def forward(self, input_ids, padding_mask):
        return self.model(input_ids, padding_mask)

    def _shared_step(self,
                     model,
                     stage,
                     batch,
                     batch_idx,
                     dataloader_idx=None):
        target = batch["target"]
        pred = model(batch['input_ids'], batch['padding_mask'])
        loss = self.criterion(pred, target)
        step_metrics = {
            f"{stage}/loss": loss,
        }
        step_output = {"loss": loss, "pred": pred, "target": target}
        return step_output, step_metrics

    def _eval_step_single_model(self,
                                model,
                                batch,
                                batch_idx,
                                dataloader_idx=None,
                                stage="val"):
        step_output, step_metrics = self._shared_step(
            model, stage, batch, batch_idx, dataloader_idx=dataloader_idx)
        # save step output
        # need .float() or .half() to deal with BFloat16 to Numpy
        pred = step_output['pred']
        if self.task_type == 'cls':
            pred = F.sigmoid(pred)
        pred = pred.half().cpu().numpy()
        target = step_output["target"].cpu().numpy()
        step_save_output = {
            "idx": batch["idx"].cpu().numpy(),
            "pred": pred,
            "target": target,
        }

        return step_output, step_save_output, step_metrics

    def training_step(self, batch, batch_idx, dataloader_idx=None):
        # self.train_epoch_idxs.extend(batch["idx"].cpu().numpy())
        return super().training_step(batch, batch_idx, dataloader_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return super().validation_step(batch, batch_idx, dataloader_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        pred = self(batch['input_ids'], batch['padding_mask'])
        if self.task_type == 'cls':
            pred = F.sigmoid(pred)
        pred = pred.float().cpu().numpy()
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
        print('start compute metric')

        idxs = [e["idx"] for e in step_outputs]
        preds = [e["pred"] for e in step_outputs]
        targets = [e["target"] for e in step_outputs]
        idxs = np.concatenate(idxs)
        orders = np.argsort(idxs)
        preds = np.concatenate(preds)[orders]
        targets = np.concatenate(targets)[orders]

        # fill nan in preds
        nan_row_idxs = np.any(np.isnan(preds), axis=-1)
        num_nan = nan_row_idxs.sum()
        if num_nan > 0:
            logger.warning("%d nan rows in prediction, set to 0", num_nan)
            preds[nan_row_idxs, :] = 0
            assert not np.any(np.isnan(preds))

        ALL_TARGETS = ['BRD4', 'HSA', 'sEH']
        target_cols = self.cfg.task.target_cols

        log_df = dataset.val_df.copy()
        log_df[[f"pred_{col}" for col in target_cols]] = preds
        log_df[[f"target_{col}" for col in target_cols]] = targets

        from src.utils.metrics import compute_metrics
        metrics = compute_metrics(log_df)

        metrics = {f"{stage}/{k}": v for k, v in metrics.items()}
        metadata = {"log_df": log_df}
        return metrics, metadata

    # @override
    def _on_epoch_end_save_metadata(self, metadata, stage, model_name):
        super()._on_epoch_end_save_metadata(metadata, stage, model_name)
        log_df = metadata["log_df"]
        df_save_path = os.path.join(
            self.cfg.env.output_metadata_dir,
            stage,
            model_name,
            f"ep={self.current_exact_epoch}_step={self.global_step}.csv",
        )
        os.makedirs(os.path.dirname(df_save_path), exist_ok=True)
        t0 = time.time()
        # log_df.to_csv(df_save_path, index=False)
        t1 = time.time()
        print('SAVE DF:', round(t1 - t0, 2), 's')

        metadata = {"log_df_path": df_save_path}

        # add `stage`/ prefix, e.g val/something
        metadata = {f"{stage}/{k}": v for k, v in metadata.items()}
        return metadata

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        metric_names = [
            'loss', 'pseudo_AP', 'nonshare_AP', 'nonshare_BRD4_AP',
            'nonshare_HSA_AP', 'nonshare_sEH_AP', 'share_AP', 'share_BRD4_AP',
            'share_HSA_AP', 'share_sEH_AP'
        ]
        metric_names = [f'val/{metric_name}' for metric_name in metric_names]
        from src.utils.misc import get_xlsx_copiable_metrics
        copiable_metrics_str = get_xlsx_copiable_metrics(
            self.metrics_tracker.all_best_metrics, metric_names)
        logger.info('COPIABLE METRICS at epoch=%f step=%d:\n%s',
                    self.current_exact_epoch, self.global_step,
                    copiable_metrics_str)

    def on_test_epoch_end(self) -> None:
        # post processing
        # calculate & log metric
        super().on_test_epoch_end()
        metric_names = [
            'loss', 'pseudo_AP', 'nonshare_AP', 'nonshare_BRD4_AP',
            'nonshare_HSA_AP', 'nonshare_sEH_AP', 'share_AP', 'share_BRD4_AP',
            'share_HSA_AP', 'share_sEH_AP'
        ]
        metric_names = [f'val/{metric_name}' for metric_name in metric_names]
        from src.utils.misc import get_xlsx_copiable_metrics
        copiable_metrics_str = get_xlsx_copiable_metrics(
            self.metrics_tracker.all_best_metrics, metric_names)
        logger.info('COPIABLE METRICS at epoch=%f step=%d:\n%s',
                    self.current_exact_epoch, self.global_step,
                    copiable_metrics_str)

    def on_predict_epoch_end(self):
        print('On predict epoch end..')
        preds = np.concatenate(self.predict_step_outputs, axis=0)
        logger.info('Prediction shape: %s', preds.shape)
        self.predict_step_outputs.clear()

        test_df = self.trainer.predict_dataloaders.dataset.test_df
        from src.utils.chem import make_submissions

        output_dir = os.path.join(
            self.cfg.env.fold_output_dir,
            getattr(self.cfg.task, 'submit_name', 'submission'))
        make_submissions(
            test_df,
            preds,
            output_dir=output_dir,
            target_cols=self.cfg.task.target_cols,
            submit_name=getattr(self.cfg.task, 'submit_name', 'submission'),
            submit_subsets=getattr(self.cfg.task, 'submit_subsets',
                                   ['all', 'share', 'public-nonshare']))

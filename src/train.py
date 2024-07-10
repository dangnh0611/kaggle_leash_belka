# this should be called at the very begining
from utils.logging import init_logging, setup_logging

init_logging()

from utils.misc import register_omegaconf_resolvers

register_omegaconf_resolvers()

import torch

torch.set_float32_matmul_precision('medium')
print('WARNING: SET float32_matmul_precision to `medium`')

from typing import Any, Dict, List, Optional, Tuple
import logging
import hydra
import lightning as ln
from lightning.pytorch.utilities import disable_possible_user_warnings
# ignore all warnings that could be false positives
disable_possible_user_warnings()
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from lightning.fabric.utilities.seed import pl_worker_init_function
import math
import copy
import os
from utils.misc import (
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    task_wrapper,
)
from utils.logging import log_hyperparameters
from data.datasets.base import BaseDataModule
from hydra.core.hydra_config import HydraConfig
import gc
from src.tasks.base import BaseTask
from pprint import pformat
from tabulate import tabulate
import json
from hydra.types import RunMode

logger = logging.getLogger(__name__)

CACHE = {}


def build_ln_datamodule(cfg, cache={}):
    logger.info("Lightning Data Module %s", cfg.data._target_)
    _cls: BaseDataModule = hydra.utils.get_class(cfg.data._target_)
    data_module = _cls(cfg, cache=cache)
    return data_module


def build_ln_module(cfg):
    logger.info("Lightning Module: %s", cfg.task._target_)
    _cls = hydra.utils.get_class(cfg.task._target_)
    module = _cls(cfg=cfg)
    return module


def build_ln_callbacks(cfg):
    ln_callbacks = instantiate_callbacks(cfg.callbacks)
    logger.info("Initialized callbacks:\n%s",
                pformat([e.__class__ for e in ln_callbacks]))
    return ln_callbacks


def build_ln_loggers(cfg):
    if not cfg.train:
        return []
    ln_loggers = instantiate_loggers(cfg.logger)
    logger.info("Initialized loggers:\n%s",
                pformat([e.__class__ for e in ln_loggers]))
    return ln_loggers


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training. This method is wrapped in optional @task_wrapper decorator, that controls the behavior
    during failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg: A DictConfig configuration composed by Hydra.
    Returns:
        A tuple with metrics and dict with all instantiated objects.
    """
    global CACHE
    # logger.info(
    #     "#################### RAW CONFIG ####################\n%s\n########################################",
    #     OmegaConf.to_yaml(cfg, resolve=False, sort_keys=False),
    # )

    # seed every things, make training deterministic
    if cfg.get("seed") is not None:
        logger.info("Seed everything with seed=%d", cfg.seed)
        seed_everything(cfg.seed, workers=True)

    ln_callbacks = build_ln_callbacks(cfg)
    ln_loggers = build_ln_loggers(cfg)

    datamodule: BaseDataModule = build_ln_datamodule(cfg, cache=CACHE)
    # if cache is empty, build new cache
    if not CACHE:
        logger.info("CACHE is empty! Start caching things..")
        CACHE = {}
        CACHE.update(datamodule.load_cache())
        # add more cache if you want
        # ...
        datamodule.set_cache(CACHE)
    logger.info("Using CACHE with keys: %s", list(CACHE.keys()))

    if cfg.train and cfg.loader.steps_per_epoch <= 0:
        # now we need to access train dataset
        datamodule.prepare_data()
        datamodule.setup(stage="fit")
        num_samples = len(datamodule.train_dataset)

        def _calculate_steps_per_epoch(cfg, num_samples):
            r"""
            Estimated stepping batches for the complete training inferred from DataLoaders, gradient
            accumulation factor and distributed setup.
            """
            num_batches = num_samples / cfg.loader.train_batch_size
            num_batches = (int(num_batches)
                           if cfg.loader.drop_last else math.ceil(num_batches))
            # This may not be accurate with small error based on how grad_accum implemented.
            # int() or math.ceil() ?
            num_steps = math.ceil(num_batches /
                                  cfg.trainer.accumulate_grad_batches)
            return num_steps

        # ceil prefer no missing samples
        actual_num_steps = _calculate_steps_per_epoch(cfg, num_samples)
        logger.warning(
            "Change steps_per_epoch=%d to %d (based on length of train dataloader)",
            cfg.loader.steps_per_epoch,
            actual_num_steps,
        )
        cfg.loader.steps_per_epoch = actual_num_steps

    logger.info(
        "#################### CONFIG ####################\n%s########################################",
        OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False),
    )

    # we should build module after data
    # so that module.configure_optimizers() workered with proper finalized config
    model: BaseTask = build_ln_module(cfg)
    if cfg.trainer.torch_compile.enable:
        model = torch.compile(model,
                              fullgraph=cfg.trainer.torch_compile.fullgraph,
                              dynamic=cfg.trainer.torch_compile.dynamic,
                              mode = cfg.trainer.torch_compile.mode)

    # setup trainer
    trainer = ln.Trainer(
        accelerator=cfg.trainer.accelerator,
        strategy=cfg.trainer.strategy,
        devices=cfg.trainer.devices,
        num_nodes=cfg.trainer.num_nodes,
        precision=cfg.trainer.precision,
        logger=ln_loggers,
        callbacks=ln_callbacks,
        fast_dev_run=cfg.trainer.fast_dev_run,
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        max_steps=cfg.trainer.max_steps,
        min_steps=cfg.trainer.min_steps,
        max_time=cfg.trainer.max_time,
        limit_train_batches=cfg.trainer.limit_train_batches,
        limit_val_batches=cfg.trainer.limit_val_batches,
        limit_test_batches=cfg.trainer.limit_test_batches,
        limit_predict_batches=cfg.trainer.limit_predict_batches,
        overfit_batches=cfg.trainer.overfit_batches,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        enable_checkpointing=cfg.trainer.enable_checkpointing,
        enable_progress_bar=cfg.trainer.enable_progress_bar,
        enable_model_summary=cfg.trainer.enable_model_summary,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        deterministic=cfg.trainer.deterministic,
        benchmark=cfg.trainer.benchmark,
        inference_mode=cfg.trainer.inference_mode,
        use_distributed_sampler=cfg.trainer.use_distributed_sampler,
        profiler=cfg.trainer.profiler,
        detect_anomaly=cfg.trainer.detect_anomaly,
        barebones=cfg.trainer.
        barebones,  # all features that may impact raw speed are disabled
        plugins=cfg.trainer.plugins,
        sync_batchnorm=cfg.trainer.sync_batchnorm,
        reload_dataloaders_every_n_epochs=cfg.trainer.
        reload_dataloaders_every_n_epochs,
        default_root_dir=cfg.trainer.default_root_dir,
    )

    if ln_loggers:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(cfg, model, datamodule, trainer)

    if cfg.train:
        logger.info("Starting training!")
        ckpt_path = cfg.get("ckpt_path", None)
        if ckpt_path is not None:
            # model.strict_loading = False
            state_dict = torch.load(ckpt_path,
                                    map_location='cpu')['state_dict']
            logger.info('Loading state dict from %s', ckpt_path)
            try:
                # @TODO: ensure key names matched
                model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                logger.warning('Error while loading state dict:\n%s', e)
                current_model_dict = model.state_dict()
                new_state_dict = {}
                ignore_keys = []
                for k, v in zip(current_model_dict.keys(),
                                state_dict.values()):
                    if current_model_dict[k].size() == v.size():
                        new_state_dict[k] = v
                    else:
                        new_state_dict[k] = current_model_dict[k]
                        ignore_keys.append(k)
                logger.warning(
                    'Ignore loading the following keys from state dict: %s',
                    ignore_keys)
                model.load_state_dict(new_state_dict, strict=False)

        trainer.fit(
            model=model,
            datamodule=datamodule,
            # ckpt_path=ckpt_path,
        )

    if cfg.test:
        logger.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            if cfg.ckpt_path:
                ckpt_path = cfg.ckpt_path
                if os.path.isdir(ckpt_path):
                    _names = os.listdir(ckpt_path)
                    if 'best.ckpt' in _names:
                        ckpt_path = os.path.join(ckpt_path, 'best.ckpt')
                    else:
                        ckpt_path = os.path.join(ckpt_path,
                                                 f'fold_{cfg.cv.fold_idx}',
                                                 'ckpts', 'best.ckpt')
                assert os.path.isfile(ckpt_path)
            else:
                logger.warning(
                    "Best ckpt not found! Using current weights for testing..."
                )
                ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        logger.info(f"Best ckpt path: {ckpt_path}")

    if cfg.predict:
        if cfg.train:
            ckpt_path = trainer.checkpoint_callback.best_model_path
        else:
            ckpt_path = cfg.ckpt_path
            assert ckpt_path is not None
            if os.path.isdir(ckpt_path):
                _names = os.listdir(ckpt_path)
                if 'best.ckpt' in _names:
                    ckpt_path = os.path.join(ckpt_path, 'best.ckpt')
                else:
                    ckpt_path = os.path.join(ckpt_path,
                                             f'fold_{cfg.cv.fold_idx}',
                                             'ckpts', 'best.ckpt')
            assert os.path.isfile(ckpt_path)

        logger.info('Loading model from %s for prediction..', ckpt_path)

        trainer.predict(model=model,
                        datamodule=datamodule,
                        ckpt_path=ckpt_path)

    all_best_metrics = model.metrics_tracker.all_best_metrics

    gc.collect()

    return all_best_metrics


@hydra.main(version_base="1.3",
            config_path="../configs/",
            config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    # extras(cfg)

    # remove lightning logger's default handler
    for name in ["lightning"]:
        target_logger = logging.getLogger(name)
        for handler in target_logger.handlers:
            target_logger.removeHandler(handler)

    # train the model
    fold_idx = cfg.cv.fold_idx
    # train all folds -> eval OOF
    if not isinstance(fold_idx, int):
        all_folds = fold_idx
    elif fold_idx == -1:
        all_folds = list(range(cfg.cv.num_folds))
    else:
        all_folds = [fold_idx]
    logger.info(
        f"Config cv.fold_idx={fold_idx}, training on {len(all_folds)} folds: {all_folds}"
    )

    fold_results = {}
    last_fold_file_handler = None

    hydra_cfg = HydraConfig.get()

    # logger.info(
    #     "HYDRA CONFIG:\n%s", OmegaConf.to_yaml(hydra_cfg, resolve=True, sort_keys=False)
    # )

    for _i, fold_idx in enumerate(all_folds):
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg.cv.fold_idx = fold_idx

        # save log for this specific fold in another log file
        fold_log_file_path = os.path.join(
            HydraConfig.get().runtime.output_dir,
            f"fold_{fold_idx}",
            f"{HydraConfig.get().job.name}.log",
        )
        if last_fold_file_handler is not None:
            # remove last file handler of previous fold
            assert isinstance(last_fold_file_handler, logging.FileHandler)
            logging.getLogger().removeHandler(last_fold_file_handler)

        last_fold_file_handler = setup_logging(
            HydraConfig.get().job_logging,
            name=None,
            file_path=fold_log_file_path,
            level=logging.getLogger().level,
        )

        logger.info("\n--------START TRAINING ON FOLD %d--------", fold_idx)
        best_metrics = train(fold_cfg)
        fold_results[fold_idx] = best_metrics

    if last_fold_file_handler is not None:
        # remove last file handler of previous fold
        assert isinstance(last_fold_file_handler, logging.FileHandler)
        logging.getLogger().removeHandler(last_fold_file_handler)

    return None


if __name__ == "__main__":
    main()

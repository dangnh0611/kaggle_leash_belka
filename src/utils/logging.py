import logging
from omegaconf import OmegaConf
from typing import Any, Dict
from lightning_utilities.core.rank_zero import rank_prefixed_message, rank_zero_only
import os


def _setup_clean_logging(name=None, level=logging.DEBUG, file_path=None):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(name)
    stream_formatter = logging.Formatter(
        fmt="\n\033[92m[%(levelname)s]: %(asctime)s %(filename)s (%(funcName)s: %(lineno)s)\n\033[0m %(message)s",
        datefmt="%H:%M:%S",
    )

    fh_formatter = logging.Formatter(
        fmt="\n[%(levelname)s]: %(asctime)s %(filename)s (%(funcName)s: %(lineno)s)\n %(message)s",
        datefmt="%H:%M:%S",
    )

    if logger.hasHandlers():
        for handler in logger.handlers:
            handler.setFormatter(stream_formatter)
    else:
        sh = logging.StreamHandler()
        sh.setFormatter(stream_formatter)
        logger.addHandler(sh)

    if file_path is not None:
        fh = logging.FileHandler(file_path)
        fh.setFormatter(fh_formatter)
        logger.addHandler(fh)

    logger.setLevel(level)
    return logger


def setup_logging(cfg, name=None, file_path=None, level=None):
    logging.basicConfig(level=logging.DEBUG)
    target_logger = logging.getLogger(name)
    file_handler = None

    if hasattr(cfg.handlers, "console"):
        console_fmt_name = cfg.handlers.console.formatter
        console_fmt = OmegaConf.to_container(getattr(cfg.formatters, console_fmt_name))
        console_fmt["fmt"] = console_fmt.pop("format")
        console_formatter = logging.Formatter(**console_fmt)

        if target_logger.hasHandlers():
            for handler in target_logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(
                    handler, logging.FileHandler
                ):
                    handler.setFormatter(console_formatter)
        else:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(console_formatter)
            target_logger.addHandler(stream_handler)

    if hasattr(cfg.handlers, "file") and file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        file_fmt_name = cfg.handlers.file.formatter
        file_fmt = OmegaConf.to_container(getattr(cfg.formatters, file_fmt_name))
        file_fmt["fmt"] = file_fmt.pop("format")
        file_formatter = logging.Formatter(**file_fmt)
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(file_formatter)
        target_logger.addHandler(file_handler)

    level = level or cfg.root.level
    target_logger.setLevel(level)
    return file_handler


def init_logging():
    """This function should be call before any loggers were initialized."""

    # Custom Logger class for Rank-based logging (useful for DDP, ..)
    class RankedLogger(logging.getLoggerClass()):
        def __init__(
            self,
            name,
            level=logging.NOTSET,
            rank_zero_only: bool = True,
        ) -> None:
            super().__init__(name, level=level)
            self.rank_zero_only = rank_zero_only

        def _log(self, level, msg, *args, rank=None, **kwargs) -> None:
            current_rank = getattr(rank_zero_only, "rank", None)
            if current_rank is None:
                raise RuntimeError(
                    "The `rank_zero_only.rank` needs to be set before use"
                )
            # msg = rank_prefixed_message(msg, current_rank)
            if self.rank_zero_only:
                if current_rank == 0:
                    return super()._log(level, msg, *args, **kwargs)
            else:
                if rank is None:
                    return super()._log(level, msg, *args, **kwargs)
                elif current_rank == rank:
                    return super()._log(level, msg, *args, **kwargs)
                else:
                    pass

    # now, register it
    old_logger_cls = logging.getLoggerClass()
    logging.setLoggerClass(RankedLogger)
    logging.warning(
        f"logging.LoggerClass had changed from {old_logger_cls} to {logging.getLoggerClass()}"
    )


@rank_zero_only
def log_hyperparameters(cfg, module, datamodule, trainer) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    del datamodule

    if not trainer.logger:
        logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams = {}
    hparams["config"] = OmegaConf.to_container(cfg)
    hparams["config_resolve"] = OmegaConf.to_container(cfg, resolve=True)

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in module.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in module.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in module.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    for logger in trainer.loggers:
        logger.log_hyperparams(hparams)

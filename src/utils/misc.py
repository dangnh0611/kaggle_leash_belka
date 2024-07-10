import warnings
from importlib.util import find_spec
from typing import Any, Callable, Dict, Optional, Tuple, List
import logging
import pprint
from tabulate import tabulate
import re
from pathlib import Path
from typing import Sequence

import rich
import rich.syntax
import rich.tree
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict

import hydra
from lightning import Callback
import math
from matplotlib import pyplot as plt
from lightning.pytorch.loggers import Logger
import os
import torch
from torch import nn
import contextlib
import numpy as np
from torch.utils.data import default_collate

logger = logging.getLogger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        logger.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(logger_cfg: DictConfig) -> List[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :return: A list of instantiated loggers.
    """
    loggers: List[Logger] = []

    if not logger_cfg:
        logger.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))

    return loggers


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.

    Utilities:
        - Ignoring python warnings
        - Setting tags from command line
        - Rich config printing

    :param cfg: A DictConfig object containing the config tree.
    """
    # return if no `extras` config
    if not cfg.get("extras"):
        logger.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        logger.info(
            "Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        logger.info(
            "Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.

    This wrapper can be used to:
        - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
        - save the exception to a `.log` file
        - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
        - etc. (adjust depending on your needs)

    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        ...
        return metric_dict, object_dict
    ```

    :param task_func: The task function to be wrapped.

    :return: The wrapped task function.
    """

    def wrap(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # execute the task
        try:
            metric_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            logger.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            logger.info(f"Output dir: {cfg.env.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    logger.info("Closing wandb!")
                    wandb.finish()

        return metric_dict

    return wrap


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    @property
    def avg(self):
        return self.sum / self.count

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n


def dict_as_table(dictionary,
                  headers=["key", "value"],
                  sort_by='key',
                  format="rounded_grid"):
    table = [[str(k), pprint.pformat(v)] for k, v in dictionary.items()]
    if sort_by is None:
        pass
    elif sort_by == "key":
        table.sort(key=lambda x: x[0])
    elif sort_by == "value":
        table.sort(key=lambda x: x[1])
    else:
        # sort_by is a function
        table.sort(key=sort_by)
    return tabulate(table, headers=headers, tablefmt=format)


@rank_zero_only
def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Prints the contents of a DictConfig as a tree structure using the Rich library.

    :param cfg: A DictConfig composed by Hydra.
    :param print_order: Determines in what order config components are printed. Default is ``("data", "model",
    "callbacks", "logger", "trainer", "paths", "extras")``.
    :param resolve: Whether to resolve reference fields of DictConfig. Default is ``False``.
    :param save_to_file: Whether to export config to the hydra output folder. Default is ``False``.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        (queue.append(field) if field in cfg else logger.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        ))

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.env.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


def register_omegaconf_resolvers():
    from omegaconf import OmegaConf
    from ast import literal_eval

    def _no_slash(s, replace="|"):
        return s.replace("/", replace)

    OmegaConf.register_new_resolver("_no_slash",
                                    _no_slash,
                                    replace=False,
                                    use_cache=False)
    OmegaConf.register_new_resolver("_eval",
                                    lambda x: eval(x),
                                    replace=False,
                                    use_cache=False)
    OmegaConf.register_new_resolver("_literal_eval",
                                    lambda x: literal_eval(x),
                                    replace=False,
                                    use_cache=False)
    OmegaConf.register_new_resolver("_or",
                                    lambda x, y: x or y,
                                    replace=False,
                                    use_cache=False)
    OmegaConf.register_new_resolver("_extend",
                                    lambda x, y: x.extend(y),
                                    replace=False,
                                    use_cache=False)
    OmegaConf.register_new_resolver(
        "_fname",
        lambda x: os.path.join(
            os.path.dirname(x),
            (os.path.basename(x)[:250] + "_ETC_"
             if len(os.path.basename(x)) > 250 else os.path.basename(x)),
        ),
        replace=False,
        use_cache=False,
    ),
    OmegaConf.register_new_resolver("_len",
                                    len,
                                    replace=False,
                                    use_cache=False)


class MetricsTracker:

    def __init__(
        self,
        metrics: Dict[str, str],
        fmt: str = "{metric}/{model_name}",
        keep_top_k: int = -1,
    ):
        # @TODO: support parse metric + model_name from regex
        assert (fmt == "{metric}/{model_name}"
                ), "Format `%s` is not supported at this moment!"
        self.metric2mode = metrics
        # assert not any(["train" in name for name in self.metric2mode.keys()
        #                 ]), "Train metric is not supported"
        # assert all([
        #     "val" in name or "test" in name
        #     for name in self.metric2mode.keys()
        # ]), "Metric name must contain `val` or `test`"
        self.fmt = fmt
        self.keep_top_k = keep_top_k
        self.all_best_metrics = {
            metric_name: []
            for metric_name in metrics.keys()
        }
        self.last_best_instance_metrics = {
            metric_name: None
            for metric_name in metrics.keys()
        }

    def _find_matched_names(self, names, key):
        fmt = self.fmt.replace("{metric}", key).replace("{model_name}", ".")
        return [
            name for name in names if re.match(fmt, name)
            if not any([name.endswith(k) for k in ["/_best_", "/_primary_"]])
        ]

    def is_equal(self, a, b, excludes=["metadata", "state"]):
        a = {k: v for k, v in a.items() if k not in excludes}
        b = {k: v for k, v in b.items() if k not in excludes}
        return a == b

    def update(
        self,
        cur_metrics: Dict[str, Any],
        cached_metadatas: Dict[str, Dict[str, Any]],
        epoch: int,
        step: int,
    ) -> None:
        for metric_name in self.all_best_metrics.keys():
            metric_mode = self.metric2mode[metric_name]
            cur_matched_metric_names = self._find_matched_names(
                cur_metrics.keys(), metric_name)
            if not cur_matched_metric_names:
                # logger.warning(
                #     "Could not found matched metric `%s` while updating metrics tracker: %s",
                #     metric_name,
                #     list(cur_metrics.keys()),
                # )
                continue
            cur_matched_metrics = [(name, cur_metrics[name])
                                   for name in cur_matched_metric_names]
            cur_matched_metrics.sort(key=lambda x: x[1],
                                     reverse=(metric_mode == "max"))
            cur_best_metric_key, cur_best_metric = cur_matched_metrics[0]
            cur_best_model_name = cur_best_metric_key.replace(
                f"{metric_name}/", "")
            assert cur_best_model_name not in ["_best_", "_primary_"]
            # update best metrics
            best_metrics = self.all_best_metrics[metric_name]
            metadata = cached_metadatas[cur_best_model_name]
            new_entry = {
                "mode": metric_mode,
                "value": cur_best_metric,
                "epoch": epoch,
                "step": step,
                "model": cur_best_model_name,
                "metadata": metadata,
                "state": cur_metrics,
            }
            old_entries = [e for e in best_metrics if e["step"] == step]
            if old_entries:
                assert len(old_entries) == 1
                old_entry = old_entries[0]
                if not self.is_equal(new_entry, old_entry):
                    logger.warning(
                        "Add new metric entry `%s` with same step %d but different value: old=%f, new=%f",
                        metric_name,
                        step,
                        old_entry["value"],
                        new_entry["value"],
                    )
                    best_metrics.append(new_entry)
                    self.last_best_instance_metrics[metric_name] = new_entry
                else:
                    # no change -> just do nothing
                    old_entry["state"].update(new_entry["state"])
                    continue
            else:
                best_metrics.append(new_entry)
                self.last_best_instance_metrics[metric_name] = new_entry
            # this implementation is more explicit
            # https://stackoverflow.com/questions/1915376/is-pythons-sorted-function-guaranteed-to-be-stable
            best_metrics.sort(
                key=lambda x: (
                    x["value"] if metric_mode == "max" else -x["value"],
                    x["epoch"],
                    x["step"],
                ),
                reverse=True,
            )
            # alternative one since python sort() is stable
            # sort by epoch, then by step, and finally by value
            # best_metrics.sort(key = lambda x: x['epoch'], reverse=True)
            # best_metrics.sort(key = lambda x: x['step'], reverse=True)
            # best_metrics.sort(key = lambda x: x['value'], reverse = (metric_mode == 'max'))

            if self.keep_top_k > 0:
                self.all_best_metrics[metric_name] = best_metrics[:self.
                                                                  keep_top_k]

    def find_top_k(self, metric_name=None):
        raise NotImplementedError

    @property
    def best_metrics(self):
        return {
            k: (v[0] if v else None)
            for k, v in self.all_best_metrics.items()
        }

    def repr_table(self, top_k=-1, fmt="rounded_grid"):
        headers = ["rank", "metric", "value", "mode", "model", "epoch", "step"]
        rows = []
        for metric_name, best_metrics in self.all_best_metrics.items():
            for i, entry in enumerate(best_metrics):
                new_row = [
                    i,
                    metric_name,
                    entry["value"],
                    entry["mode"],
                    entry["model"],
                    entry["epoch"],
                    entry["step"],
                ]
                rows.append(new_row)
                if top_k > 0 and i >= top_k:
                    break
        table = tabulate(rows, headers=headers, tablefmt=fmt)
        return table


def get_subplots(n, nrows=None, ncols=None, figsize=None):
    if nrows:
        ncols = math.ceil(n / nrows)
    elif ncols:
        nrows = math.ceil(n / ncols)
    else:
        raise ValueError

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.tight_layout()
    return fig, axes.flat


def tf_init_weights(m):
    """
    Pytorch may have bad default weights initialisation.
    Ref:
        https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
        https://www.kaggle.com/competitions/liverpool-ion-switching/discussion/145256#863764
        https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
        https://discuss.pytorch.org/t/suboptimal-convergence-when-compared-with-tensorflow-model/5099/52
    
    """
    classname = m.__class__.__name__
    if 'conv' in classname.lower():
        if hasattr(m, 'weight'):
            nn.init.xavier_uniform_(m.weight,
                                    gain=nn.init.calculate_gain('relu'))
        if hasattr(m, 'bias'):
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if hasattr(m, 'bias'):
            nn.init.constant_(m.bias, 0.0)


def small_init_embed(m, a=-1e-4, b=1e-4):
    """Ref: https://github.com/BlinkDL/SmallInitEmb"""
    if isinstance(m, (nn.Embedding)):
        logger.info(
            'Apply SmallInitEmbed (https://github.com/BlinkDL/SmallInitEmb) to %s',
            m)
        nn.init.uniform_(m.weight, a=a, b=b)  # SmallInit(Emb)


@contextlib.contextmanager
def local_numpy_temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def padding_collater(samples,
                     max_len,
                     keys=['tokens', 'padding_mask'],
                     pad_values=[0, 0],
                     padding=True):
    """
    This assume that each Tensor in `keys` has same length at T_dim=0
    """
    first_key = keys[0]
    bs = len(samples)
    sample_lens = [e[first_key].size(0) for e in samples]
    if len(set(sample_lens)) == 1:
        # all has same len, use default collater
        return default_collate(samples)

    padding_samples = [{k: sample.pop(k) for k in keys} for sample in samples]

    batch = default_collate(samples)

    # padding or truncating
    if padding:
        target_len = min(max(sample_lens), max_len)
    else:
        target_len = min(min(sample_lens), max_len)

    for k, pad_v in zip(keys, pad_values):
        batch[k] = padding_samples[0][k].new_full(
            (bs, target_len, *padding_samples[0][k].shape[1:]), pad_v)

    for i, (sample, sample_len) in enumerate(zip(padding_samples,
                                                 sample_lens)):
        diff = target_len - sample_len
        if diff == 0:
            for k in keys:
                batch[k][i] = sample[k]
        elif diff > 0:
            assert padding
            for k in keys:
                batch[k][i, :sample_len] = sample[k]
        else:
            # truncate
            for k in keys:
                batch[k][i] = sample[k][:max_len]

    return batch


def get_xlsx_copiable_metrics(best_metrics,
                              metric_names=['val/loss'],
                              value_round=6,
                              epoch_round=1):
    cells = []
    for name in metric_names:
        if name in best_metrics:
            best_entry = best_metrics[name]
            if best_entry:
                best_entry = best_entry[0]
                cell = f'{round(best_entry["value"], value_round)} ({round(best_entry["epoch"], epoch_round)})'
            else:
                cell = 'N/A'
        else:
            cell = 'N/A'
        cells.append(cell)
    header_str = '	'.join(metric_names)
    cell_str = '	'.join(cells)
    ret_str = header_str + '\n' + cell_str
    return ret_str


class MLMMasker:
    """
    Ref: https://github.com/huggingface/transformers/blob/0dd65a03198424a41ec6948e445c313e9f292939/src/transformers/data/data_collator.py#L827
    """

    def __init__(self,
                 tokenizer,
                 mlm_prob=0.15,
                 mask_prob=0.8,
                 random_prob=0.1,
                 ):
        self.tokenizer = tokenizer
        self.mlm_prob = mlm_prob
        self.mask_prob = mask_prob
        self.random_prob = random_prob
        self._random_prob = random_prob / (1. - mask_prob)
        self.mask_token_id = tokenizer.mask_token_id

    def __repr__(self):
        return f'MLMMasker(tokenizer={self.tokenizer.__class__}, mlm_prob = {self.mlm_prob}, mask_prob = {self.mask_prob}, random_prob = {self.random_prob}, mask_token_id = {self.mask_token_id})'

    def __call__(self, inputs, special_tokens_mask=None):
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_prob)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True)
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask,
                                               dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -1  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(
            torch.full(labels.shape, self.mask_prob)).bool() & masked_indices
        inputs[indices_replaced] = self.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(
            torch.full(
                labels.shape,
                self._random_prob)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer),
                                     labels.shape,
                                     dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class dotdict(dict):
	__setattr__ = dict.__setitem__
	__delattr__ = dict.__delitem__
	
	def __getattr__(self, name):
		try:
			return self[name]
		except KeyError:
			raise AttributeError(name)
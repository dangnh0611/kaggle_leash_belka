import numpy as np
import pandas as pd
from typing import Optional, Union
from sklearn.metrics import average_precision_score
import logging
import itertools
from tqdm import tqdm

logger = logging.getLogger(__name__)

SUBSETS = ['nonshare', 'share', 'share1', 'share2']
PROTEINS = ['BRD4', 'HSA', 'sEH']


def compute_metrics(df, stage='test'):
    df = df.reset_index(drop=True)
    target_cols = [f'target_{protein}' for protein in PROTEINS if f'target_{protein}' in df.columns]
    df[target_cols] = (df[target_cols] > 0.5).astype('uint8')
    all_metrics = {}
    
    logger.info('Computing metrics..')
    for subset in SUBSETS:
        if subset == 'nonshare':
            subset_df = df[df.subset == 0].reset_index(drop=True)
        elif subset == 'share':
            subset_df = df[df.subset != 0].reset_index(drop=True)
        elif subset == 'share1':
            subset_df = df[df.subset == 1].reset_index(drop=True)
        elif subset == 'share2':
            subset_df = df[df.subset.isin([2, 3])].reset_index(drop=True)
        else:
            raise AssertionError

        for protein in PROTEINS:
            filter_name = f'{subset}_{protein}'
            # compute metrics
            metrics = {}
            if len(subset_df
                   ) == 0 or f'target_{protein}' not in subset_df.columns:
                logger.warning(
                    '[METRICS] Skip subset=%s protein=%s and set metric value to None..',
                    subset, protein)
                metrics['AP'] = None
            else:
                preds = subset_df[f'pred_{protein}']
                targets = subset_df[f'target_{protein}']
                metrics['AP'] = average_precision_score(targets,
                                                        preds,
                                                        pos_label=1,
                                                        average='micro')

            # prefix with filter name
            for k, v in metrics.items():
                metric_name = f"{filter_name}_{k}" if filter_name != "" else k
                all_metrics[metric_name] = v

    all_metrics['pseudo_AP'] = np.mean([
        all_metrics[k] for k in [
            f'{subset}_{protein}_AP' for subset, protein in list(
                itertools.product(['nonshare', 'share2'], PROTEINS))
        ] if all_metrics[k] is not None
    ])
    all_metrics['AP'] = np.mean([
        all_metrics[k] for k in [
            f'{subset}_{protein}_AP' for subset, protein in list(
                itertools.product(['nonshare', 'share'], PROTEINS))
        ] if all_metrics[k] is not None
    ])
    all_metrics['nonshare_AP'] = np.mean([
        all_metrics[k] for k in [
            f'{subset}_{protein}_AP' for subset, protein in list(
                itertools.product(['nonshare'], PROTEINS))
        ] if all_metrics[k] is not None
    ])
    all_metrics['share_AP'] = np.mean([
        all_metrics[k] for k in [
            f'{subset}_{protein}_AP'
            for subset, protein in list(itertools.product(['share'], PROTEINS))
        ] if all_metrics[k] is not None
    ])
    all_metrics['share1_AP'] = np.mean([
        all_metrics[k] for k in [
            f'{subset}_{protein}_AP' for subset, protein in list(
                itertools.product(['share1'], PROTEINS))
        ] if all_metrics[k] is not None
    ])
    all_metrics['share2_AP'] = np.mean([
        all_metrics[k] for k in [
            f'{subset}_{protein}_AP' for subset, protein in list(
                itertools.product(['share2'], PROTEINS))
        ] if all_metrics[k] is not None
    ])

    # remove non-exist metrics
    all_metrics = {
        k: v
        for k, v in all_metrics.items() if v is not None and not np.isnan(v)
    }
    return all_metrics

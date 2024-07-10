from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import LearningRateFinder, ModelCheckpoint
from lightning.pytorch.callbacks.progress import TQDMProgressBar
import os
from matplotlib import pyplot as plt
import logging

logger = logging.getLogger(__name__)


class VerboseLearningRateFinder(LearningRateFinder):
    def __init__(self, save_dir, milestones, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_dir
        self.milestones = milestones

    def on_fit_start(self, *args, **kwargs):
        return

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in self.milestones or trainer.current_epoch == 0:
            self.lr_find(trainer, pl_module)
            # plot
            fig = self.optimal_lr.plot()
            plt.savefig(os.path.join(self.save_dir, f'lr_finder_ep={trainer.current_epoch}.png'))
            plt.close()
            suggested_lr = self.optimal_lr.suggestion()
            logger.info('Best learning rate suggestion %f at epoch %d', suggested_lr, trainer.current_epoch)


class CustomTQDMProgressBar(TQDMProgressBar):
    BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}{postfix}]"

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        super().on_validation_epoch_end(trainer, pl_module)
        # print new lines so that next call to print() or log()
        # won't start on the same line as progress bar
        # -> just look more clean :)
        print('\n\n')

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        return super().on_train_epoch_end(trainer, pl_module)

    def get_metrics(self, *args, **kwargs):
        # don't show the version number
        items = super().get_metrics(*args, **kwargs)
        items.pop("v_num", None)
        return items


class CustomModelCheckpoint(ModelCheckpoint):
    
    def _update_best_and_save(self, current, trainer, monitor_candidates) -> None:
        super()._update_best_and_save(current, trainer, monitor_candidates)
        best_model_path = self.best_model_path
        best_score = self.best_model_score
        best_symlink_path = os.path.join(self.dirpath, 'best.ckpt')
        self._link_checkpoint(trainer, best_model_path, best_symlink_path)


from timm.scheduler.scheduler_factory import create_scheduler_v2
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION
import torch
import logging


logger = logging.getLogger(__name__)

TORCH_SCHEDULERS = {
    "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
    "onecycle": torch.optim.lr_scheduler.OneCycleLR,
}


def resolve_scheduler_kwargs(cfg):
    name = cfg.name
    kwargs = {"name": name}

    min_lr = cfg.lr * cfg.min_lr_factor
    warmup_lr = cfg.lr * cfg.warmup_lr_factor
    total_steps = int(cfg.max_epochs * cfg.steps_per_epoch)
    total_sched_steps = int(
        (cfg.max_epochs - cfg.cooldown_epochs) * cfg.steps_per_epoch
    )
    warmup_steps = int(cfg.warmup_epochs * cfg.steps_per_epoch)

    # default is to update lr per step
    interval = "step"

    OVERRIDE_KWARGS_DICT = {
        "transformers@cosine_with_restarts": dict(
            num_warmup_steps=warmup_steps,
            num_training_steps=total_sched_steps,
            num_cycles=cfg.cycle_limit,
            last_epoch=-1,
        ),
        "transformers@linear": dict(
            num_warmup_steps=warmup_steps,
            num_training_steps=total_sched_steps,
            last_epoch=-1,
        ),
        "transformers@cosine": dict(
            num_warmup_steps=warmup_steps,
            num_training_steps=total_sched_steps,
            last_epoch=-1,
        ),
        "transformers@polynomial": dict(
            num_warmup_steps=warmup_steps,
            num_training_steps=total_sched_steps,
            lr_end=min_lr,
            power=1.0,
            last_epoch=-1,
        ),
        "timm@cosine": dict(
            num_epochs=cfg.max_epochs - cfg.cooldown_epochs,
            decay_epochs=5,
            decay_milestones=(12, 24, 36),
            cooldown_epochs=cfg.cooldown_epochs,
            patience_epochs=5,
            decay_rate=0.1,
            min_lr=min_lr,
            warmup_lr=warmup_lr,
            warmup_epochs=cfg.warmup_epochs,
            warmup_prefix=False,
            noise=None,
            noise_pct=0.67,
            noise_std=1.0,
            noise_seed=42,
            cycle_mul=1.0,
            cycle_decay=cfg.cycle_decay,
            cycle_limit=cfg.cycle_limit,
            k_decay=1.0,
            plateau_mode=cfg.metric_mode,
            step_on_epochs=False,
            updates_per_epoch=cfg.steps_per_epoch,
        ),
        "torch@cosine": dict(T_max=total_steps, eta_min=min_lr, verbose=False),
        "torch@onecycle": dict(
            max_lr=getattr(cfg, 'max_lr', None) or cfg.lr,
            epochs=cfg.max_epochs - cfg.cooldown_epochs,
            steps_per_epoch=cfg.steps_per_epoch,
            pct_start=getattr(cfg, "pct_start", None)
            or cfg.warmup_epochs / (cfg.max_epochs - cfg.cooldown_epochs),
            anneal_strategy=getattr(cfg, "anneal_strategy", "cos"),
            cycle_momentum=getattr(cfg, "cycle_momentum", True),
            base_momentum=getattr(cfg, "base_momentum", 0.85),
            max_momentum=getattr(cfg, "max_momentum", 0.95),
            div_factor=getattr(cfg, "div_factor", None) or 1.0 / cfg.warmup_lr_factor,
            final_div_factor=getattr(cfg, "final_div_factor", None)
            or cfg.warmup_lr_factor / cfg.min_lr_factor,
            three_phase=getattr(cfg, "three_phase", False),
            last_epoch=-1,
        ),
    }

    override_kwargs = OVERRIDE_KWARGS_DICT[name]
    kwargs.update(override_kwargs)

    return kwargs, interval


# @TODO: support cooldown in transformers's scheduler
# simply by setting a min_lr value instead of 0
def create_scheduler(optimizer, cfg):
    sched_kwargs, interval = resolve_scheduler_kwargs(cfg)
    sched_name = sched_kwargs.pop("name")
    logger.info('Scheduler kwargs: %s', sched_kwargs)

    if sched_name.startswith("transformers@"):
        sched_name = sched_name.replace("transformers@", "")
        sched_func = TYPE_TO_SCHEDULER_FUNCTION[sched_name]
        scheduler = sched_func(optimizer, **sched_kwargs)
    elif sched_name.startswith("torch@"):
        sched_name = sched_name.replace("torch@", "")
        sched_func = TORCH_SCHEDULERS[sched_name]
        scheduler = sched_func(optimizer, **sched_kwargs)
    elif sched_name.startswith("timm@"):
        sched_name = sched_name.replace("timm@", "")
        scheduler, num_epochs = create_scheduler_v2(
            optimizer, sched=sched_name, **sched_kwargs
        )
        logger.info("Timm scheduler %s with num_epochs=%f", scheduler, num_epochs)
    else:
        raise NotImplementedError
    return {
        "scheduler": scheduler,
        "interval": interval,
        "reduce_on_plateau": sched_kwargs.get("reduce_on_plateau", False),
        "monitor": cfg.metric,
    }

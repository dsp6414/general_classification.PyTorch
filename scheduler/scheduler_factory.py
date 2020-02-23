# coding : utf-8

from .cosine_lr import CosineLRScheduler
from .tanh_lr import TanhLRScheduler
from .step_lr import StepLRScheduler


def create_scheduler(cfg, optimizer):
    num_epochs = cfg.SOLVER.MAX_EPOCHS
    scheduler = cfg.SOLVER.SCHEDULER
    lr_scheduler = None
    #FIXME expose cycle parms of the scheduler config to arguments
    if scheduler == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=cfg.SOLVER.MIN_LR,
            decay_rate=cfg.SOLVER.DECAY_RATE,
            warmup_lr_init=cfg.SOLVER.WARMUP_LR,
            warmup_t=cfg.SOLVER.WARMUP_EPOCHS,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cfg.SOLVER.COOLDOWN_EPOCHS
    elif scheduler == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=num_epochs,
            t_mul=1.0,
            lr_min=cfg.SOLVER.MIN_LR,
            warmup_lr_init=cfg.SOLVER.WARMUP_LR,
            warmup_t=cfg.SOLVER.WARMUP_EPOCHS,
            cycle_limit=1,
            t_in_epochs=True,
        )
        num_epochs = lr_scheduler.get_cycle_length() + cfg.SOLVER.COOLDOWN_EPOCHS
    elif scheduler == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=cfg.SOLVER.DECAY_EPOCHS,
            decay_rate=cfg.SOLVER.DECAY_RATE,
            warmup_lr_init=cfg.SOLVER.WARMUP_LR,
            warmup_t=cfg.SOLVER.WARMUP_EPOCHS,
        )
    return lr_scheduler, num_epochs

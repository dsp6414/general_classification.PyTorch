# coding : utf-8

import torch
from torch import optim as optim
from optim import Nadam, RMSpropTF, AdamW, RAdam, NovoGrad, NvNovoGrad, Lookahead


def create_optimizer(cfg, model, filter_bias_and_bn=True):
    opt_lower = cfg.SOLVER.OPTIMIZER.lower()
    lr = cfg.SOLVER.BASE_LR
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    momentum = cfg.SOLVER.MOMENTUM
    if 'adamw' in opt_lower or 'radam' in opt_lower:
        # Compensate for the way current AdamW and RAdam optimizers apply LR to the weight-decay
        # I don't believe they follow the paper or original Torch7 impl which schedules weight
        # decay based on the ratio of current_lr/initial_lr
        weight_decay /= lr
    if weight_decay and filter_bias_and_bn:
        parameters = []
        for key, value in model.named_parameters():
            if not value.requires_grad:
                continue
            filtered_lr = lr
            filtered_weight_decay = weight_decay
            if "bias" in key:
                filtered_lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
                filtered_weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
            parameters += [{"params": [value], "lr": filtered_lr, "weight_decay": filtered_weight_decay}]
        weight_decay = 0.
    else:
        parameters = model.parameters()

    opt_split = opt_lower.split('_')
    opt_name = opt_split[-1]
    if opt_name == 'sgd':
        optimizer = optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)
    elif opt_name == 'adam':
        optimizer = optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        optimizer = AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'nadam':
        optimizer = Nadam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'radam':
        optimizer = RAdam(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adadelta':
        optimizer = optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        optimizer = optim.RMSprop(parameters, lr=lr, alpha=0.9, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == 'rmsproptf':
        optimizer = RMSpropTF(parameters, lr=lr, alpha=0.9, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == 'novograd':
        optimizer = NovoGrad(parameters, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'nvnovograd':
        optimizer = NvNovoGrad(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("Invalid optimizer")

    if len(opt_split) > 1:
        if opt_split[0] == 'lookahead':
            optimizer = Lookahead(optimizer)
    return optimizer

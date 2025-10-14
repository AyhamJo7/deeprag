from typing import Optional

import torch
from torch.optim import AdamW
from transformers import (SchedulerType, get_scheduler)


def get_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> AdamW:
    """Creates an AdamW optimizer with weight decay for the given model."""
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    return AdamW(optimizer_grouped_parameters, lr=lr)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    scheduler_type: SchedulerType = SchedulerType.LINEAR,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Creates a learning rate scheduler with a warmup period."""
    return get_scheduler(
        name=scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

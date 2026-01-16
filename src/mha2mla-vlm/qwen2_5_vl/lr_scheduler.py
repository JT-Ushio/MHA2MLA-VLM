import math
import logging
from types import SimpleNamespace
from functools import partial

import torch
from swift.trainers import TrainingArguments
from swift.utils import get_dist_setting
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional

logger = logging.getLogger(__name__)

optimizer_lr_scheduler_kwargs: Optional[dict] = None 

def set_lr_scheduler_args(args: dict | str):
    global optimizer_lr_scheduler_kwargs
    if type(args) is str:
        import json
        args = json.loads(args)
    
    # Define required parameters
    required_params = [
        'lr_warmup_steps',
        'lr_warmup_style',
        'lr_decay_starting_step',
        'lr_decay_steps',
        'lr_decay_style',
        'min_decay_lr'
    ]
    
    # Check for missing parameters
    missing_params = [param for param in required_params if param not in args]
    if missing_params:
        raise ValueError(
            f"Missing required optimizer_lr_scheduler_kwargs parameters: {missing_params}\n"
            f"Required parameters are: {required_params}\n"
            f"Please ensure all parameters are specified in your configuration file."
        )
    
    # Validate numeric parameters are non-negative
    numeric_params = ['lr_warmup_steps', 'lr_decay_starting_step', 'lr_decay_steps', 'min_decay_lr']
    for param in numeric_params:
        if args[param] is not None and args[param] < 0:
            raise ValueError(
                f"Parameter '{param}' must be non-negative, got {args[param]}"
            )
    
    # Validate style parameters are not None
    if args['lr_warmup_style'] is None:
        raise ValueError("Parameter 'lr_warmup_style' must not be None")
    
    if args['lr_decay_style'] is None:
        raise ValueError("Parameter 'lr_decay_style' must not be None")
    
    optimizer_lr_scheduler_kwargs = args
    logger.info(f"✓ LR scheduler arguments validated successfully with keys: {list(args.keys())}")


def log_lr_scheduler_args():
    global optimizer_lr_scheduler_kwargs
    print(f"Optimizer and LR Scheduler Arguments: ${optimizer_lr_scheduler_kwargs}")

def calculate_max_steps(args: 'TrainingArguments', dataset) -> int:
    if args.max_steps and args.max_steps > 0:
        max_steps = args.max_steps
    else:
        len_dataset = len(dataset)
        _, _, world_size, _ = get_dist_setting()
        total_train_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps * world_size
        num_update_steps_per_epoch = len_dataset // total_train_batch_size
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
        max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
    return max_steps

def create_constant_with_warmup_decay_optimizer(train_args: 'TrainingArguments', model, dataset):
    """
    Create a constant with warmup decay optimizer.
    """
    global optimizer_lr_scheduler_kwargs
    optimizer_name = train_args.optim
    max_steps = calculate_max_steps(train_args, dataset)
    if "adam" in optimizer_name:
        optimizer = torch.optim.AdamW(
            params=model.parameters(),
            lr=train_args.learning_rate,
            betas=(
                train_args.adam_beta1,
                train_args.adam_beta2,
            ),
            eps=train_args.adam_epsilon,
            weight_decay=train_args.weight_decay,
            fused=bool(train_args.optim == "adamw_torch_fused"),
        )
    else:
        raise ValueError(f"Unknown optimizer factory {optimizer_name}")
    
    lr_scheduler = lr_scheduler_builder(
        optimizer=optimizer,
        lr_scheduler_args=SimpleNamespace(**optimizer_lr_scheduler_kwargs),
        total_training_steps=max_steps,
    )
    return optimizer, lr_scheduler


def lr_scheduler_builder(
    optimizer: Optimizer, lr_scheduler_args, total_training_steps: int
):
    if lr_scheduler_args.lr_decay_steps is None:
        lr_decay_steps = total_training_steps
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_steps -= lr_scheduler_args.lr_warmup_steps
        if lr_scheduler_args.lr_decay_starting_step is not None:
            lr_decay_steps -= lr_scheduler_args.lr_decay_starting_step
    else:
        lr_decay_steps = lr_scheduler_args.lr_decay_steps

    if lr_scheduler_args.lr_decay_starting_step is None:
        if lr_scheduler_args.lr_warmup_steps is not None:
            lr_decay_starting_step = lr_scheduler_args.lr_warmup_steps
        else:
            lr_decay_starting_step = 0
    else:
        lr_decay_starting_step = lr_scheduler_args.lr_decay_starting_step

    print("[DEBUG]: lr_scheduler_args: ", lr_scheduler_args)
    

    def lr_lambda(current_step: int, initial_lr: float):
        """
        current_step: current training step
        initial_lr: the learning rate of a parameter group

        More info on initial_lr:
        And in standard parameterization, lr_lambda only takes a single learning rate.
        But in µTransfer, each parameter has a custom learning rate (custom_lr = lr_scheduler_args.learning_rate * scaling_factor),
        so each parameter group has a custom lr_lambda function.

        LR Scheduling function, it has from 2 up to 4 phases:
        - warmup,
        - optional: constant (if lr_decay_starting_step is set)
        - decay
        - optional: constant (if lr_decay_steps and/or lr_decay_starting_step are set)
        Warmup starts at lr=0 and ends at `lr=lr`
        Then it stays constant at lr if lr_decay_starting_step is set and larger than lr_warmup_steps
        Then it decays until `min_decay_lr` for lr_decay_steps if set, else: (total_training_steps - lr_warmup_steps or lr_decay_starting_step)
        Then it stays constant at min_decay_lr if lr_decay_starting_step is set and total_training_steps is larger)
        """
        # No warmup or decay
        if lr_scheduler_args.lr_warmup_steps == 0 and lr_decay_steps == 0:
            return initial_lr

        # Warmup phase
        elif (
            lr_scheduler_args.lr_warmup_style is not None
            and current_step <= lr_scheduler_args.lr_warmup_steps
        ):
            if lr_scheduler_args.lr_warmup_style == "linear":
                lmbda = (
                    initial_lr
                    * current_step
                    / max(lr_scheduler_args.lr_warmup_steps, 1)
                )
            elif lr_scheduler_args.lr_warmup_style == "constant":
                lmbda = lr_scheduler_args.learning_rate
            else:
                raise ValueError(
                    f"Unknown warmup style {lr_scheduler_args.lr_warmup_style}"
                )

        # Optional constant phase at learning_rate
        elif current_step < lr_decay_starting_step:
            lmbda = initial_lr

        # Decay phase
        elif (
            lr_scheduler_args.lr_decay_style is not None
            and current_step < lr_decay_starting_step + lr_decay_steps
        ):
            if lr_scheduler_args.lr_decay_style == "cosine":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (current_step - lr_decay_starting_step)
                            / lr_decay_steps
                        )
                    )
                    / 2
                )
            elif lr_scheduler_args.lr_decay_style == "linear":
                lmbda = (
                    lr_scheduler_args.min_decay_lr
                    + (initial_lr - lr_scheduler_args.min_decay_lr)
                    * (lr_decay_steps - (current_step - lr_decay_starting_step))
                    / lr_decay_steps
                )
            elif lr_scheduler_args.lr_decay_style == "1-sqrt":
                lmbda = lr_scheduler_args.min_decay_lr + (
                    initial_lr - lr_scheduler_args.min_decay_lr
                ) * (
                    1
                    - math.sqrt(
                        (current_step - lr_decay_starting_step) / lr_decay_steps
                    )
                )
            else:
                raise ValueError(
                    f"Unknown decay style {lr_scheduler_args.lr_decay_style}"
                )

        # Optional constant phase at min_decay_lr
        else:
            lmbda = lr_scheduler_args.min_decay_lr

        lmbda /= initial_lr  # Normalization for pytorch
        return lmbda

    def get_lr_lambda_for_param_group(lr: float):
        return partial(lr_lambda, initial_lr=lr)

    # NOTE: get learning rate scheduler for each param group
    # NOTE: Changes made.
    lr_lambdas = []
    for param_group in optimizer.param_groups:
        lr_lambdas.append(get_lr_lambda_for_param_group(lr=param_group["lr"]))

    assert len(lr_lambdas) == len(optimizer.param_groups), (
        "Custom learning rate functions dont match the number of param groups"
    )

    logger.info(
        f"[Optimizer Building] There are total {len(lr_lambdas)} custom learning rate function for parameter groups",
        logger=logger,
        level=logging.DEBUG,
    )

    lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambdas)
    return lr_scheduler

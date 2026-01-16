"""
Common utility functions for MHA2MLA training.

This module provides shared utility functions used across different training scripts
and stages, reducing code duplication and improving maintainability.
"""
import logging

logger = logging.getLogger(__name__)


def convert_ratio_to_steps(lr_scheduler_kwargs, total_steps):
    """
    Convert ratio-based parameters (0.0-1.0) to absolute step numbers.
    
    This function processes learning rate scheduler parameters and converts any values
    between 0.0 and 1.0 to absolute step numbers based on the total training steps.
    Values greater than 1 are assumed to be absolute steps and remain unchanged.
    
    Args:
        lr_scheduler_kwargs (dict): Dictionary containing scheduler parameters
        total_steps (int): Total number of training steps
        
    Returns:
        dict: New dictionary with converted step values
        
    Example:
        >>> convert_ratio_to_steps({'lr_warmup_steps': 0.1}, 1000)
        {'lr_warmup_steps': 100}
    """
    if not lr_scheduler_kwargs:
        return lr_scheduler_kwargs
        
    converted_kwargs = {}
    step_params = ['lr_warmup_steps', 'lr_decay_starting_step', 'lr_decay_steps']
    
    for key, value in lr_scheduler_kwargs.items():
        if key in step_params and isinstance(value, (int, float)):
            if 0.0 <= value <= 1.0:
                # Convert ratio to absolute steps
                converted_value = int(value * total_steps)
                converted_kwargs[key] = converted_value
                logger.info(f"[LR Scheduler] Converted {key}: {value} ({value*100:.1f}%) -> {converted_value} steps")
            else:
                # Keep absolute step value
                converted_kwargs[key] = int(value)
                logger.info(f"[LR Scheduler] Using absolute {key}: {int(value)} steps")
        else:
            # Keep non-step parameters unchanged
            converted_kwargs[key] = value
            
    return converted_kwargs


def freeze_non_attn_weights(model, version):
    print(f"......freeze_non_attn_weights version: {version}......")
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        flag = False
        if (
            # V2: self_attn & layer_norm
            (version=="v2" and ("attn" in name or "norm" in name))
            # V3: q_proj, up_v, up_k, k_r_proj, down_kv
            or (version=="v3" and "attn" in name and "o_proj" not in name)
            # V4: q_proj, up_v, up_k, down_kv
            or (version=="v4" and "attn" in name and "o_proj" not in name and "k_r_proj" not in name or ("norm" in name))
            # V5: up_v, up_k, down_kv
            or (version=="v5" and "attn" in name and "o_proj" not in name and "k_r_proj" not in name and "q_proj" not in name)
            # V5: up_v, up_k, down_kv
            or (version=="v6" and "attn" in name and "o_proj" not in name and "k_r_proj" not in name and "q_proj" not in name and 'up_' not in name)
        ):
            flag = True
        param.requires_grad = flag

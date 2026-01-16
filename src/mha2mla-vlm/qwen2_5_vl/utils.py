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
    
    This function allows using percentages in scheduler configuration for better
    adaptability across different training setups (batch size, GPU count, etc.).
    
    Args:
        lr_scheduler_kwargs: Dictionary with scheduler configuration.
                            Values between 0 and 1 are treated as ratios.
        total_steps: Total number of training steps
        
    Returns:
        dict: Modified dictionary with ratios converted to absolute steps
        
    Examples:
        >>> kwargs = {
        ...     'lr_warmup_steps': 0.1,      # 10% of total steps
        ...     'lr_decay_steps': 0.1,       # 10% of total steps
        ... }
        >>> convert_ratio_to_steps(kwargs, 1000)
        {'lr_warmup_steps': 100, 'lr_decay_steps': 100}
        
        >>> kwargs = {
        ...     'lr_warmup_steps': 100,      # Absolute value (>1)
        ...     'lr_decay_steps': 0.05,      # Ratio (0-1)
        ... }
        >>> convert_ratio_to_steps(kwargs, 1000)
        {'lr_warmup_steps': 100, 'lr_decay_steps': 50}
    """
    if not isinstance(lr_scheduler_kwargs, dict):
        return lr_scheduler_kwargs
    
    converted_kwargs = lr_scheduler_kwargs.copy()
    
    # Parameters that support ratio conversion
    ratio_params = ['lr_warmup_steps', 'lr_decay_starting_step', 'lr_decay_steps']
    
    for param in ratio_params:
        if param in converted_kwargs and converted_kwargs[param] is not None:
            value = converted_kwargs[param]
            # If value is between 0.0 and 1.0 (inclusive), treat as ratio
            if 0.0 <= value <= 1.0:
                converted_value = int(total_steps * value)
                logger.info(
                    f"[Scheduler Config] {param}: {value:.2%} → {converted_value} steps "
                    f"(total: {total_steps})"
                )
                converted_kwargs[param] = converted_value
            else:
                logger.info(f"[Scheduler Config] {param}: {int(value)} steps (absolute)")
    
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

"""
Custom template hooks for Qwen2.5-VL model preprocessing.

This module provides custom pre-forward and post-encode hooks for Qwen2-VL
template processing, especially for handling multimodal inputs (images/videos)
during training with DeepSpeed.
"""

import inspect
from typing import Dict, Any

import torch
from torch import nn
from peft import PeftModel
from swift.llm import to_device
from swift.utils import is_deepspeed_enabled
from swift.llm.template.base import Template
from swift.llm.template.template.qwen import Qwen2VLTemplate


def custom_pre_forward_hook(self, model: nn.Module, args, kwargs):
    """
    Custom pre-forward hook for template processing.
    
    This hook processes the inputs before forward pass, handling device placement
    and cleaning up incompatible parameters based on model signature.
    
    Args:
        self: Template instance
        model: The model to forward
        args: Positional arguments
        kwargs: Keyword arguments containing input_ids, attention_mask, etc.
    
    Returns:
        Tuple of processed (args, kwargs)
    """
    old_kwargs = to_device(kwargs, model.device)
    kwargs = to_device(self._post_encode(model, old_kwargs), model.device)
    
    # Preserve essential keys from original kwargs
    for k, v in old_kwargs.items():
        if (
            k in {"input_ids", "attention_mask", "labels", "position_ids"}
            and k not in kwargs
        ):
            kwargs[k] = v

    # Remove position_ids if not supported by the model
    if isinstance(model, PeftModel):
        parameters = inspect.signature(model.model.forward).parameters
    else:
        parameters = inspect.signature(model.forward).parameters
    if "position_ids" not in parameters:
        kwargs.pop("position_ids", None)
    
    return args, kwargs


def custom_post_encode(self, model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Custom post-encode function for Qwen2-VL multimodal processing.
    
    This function handles the fusion of text embeddings with image/video embeddings,
    supporting both plain-text and multimodal inputs. For DeepSpeed training with
    plain-text, it adds a dummy visual input to avoid computation graph issues.
    
    Args:
        self: Qwen2VLTemplate instance
        model: The Qwen2-VL model
        inputs: Dictionary containing input_ids, pixel_values, etc.
    
    Returns:
        Dictionary with input_ids and processed inputs_embeds
    """
    if not self.is_training:
        return inputs
    
    input_ids = inputs["input_ids"]
    
    # Get the model's embedding layer (handle LoRA case)
    _model = model.model
    if not hasattr(_model, "embed_tokens"):
        _model = _model.model  # LoRA wrapped model
    
    # Extract multimodal inputs
    pixel_values = inputs.get("pixel_values")
    pixel_values_videos = inputs.get("pixel_values_videos")
    image_grid_thw = inputs.get("image_grid_thw")
    video_grid_thw = inputs.get("video_grid_thw")

    # Get text embeddings
    inputs_embeds = _model.embed_tokens(input_ids)

    # Determine the dtype for visual inputs
    dtype = model.visual.get_dtype() if self.version == "v2" else model.visual.dtype
    
    # Handle plain-text case (no images/videos)
    if pixel_values is None and pixel_values_videos is None:
        if is_deepspeed_enabled():
            # Create a dummy image to maintain computation graph in DeepSpeed
            from PIL import Image

            images = [Image.new("RGB", (32, 32), (0, 0, 0))]
            media_inputs = self.processor.image_processor(
                images=images, videos=None, return_tensors="pt"
            )
            device = input_ids.device
            media_inputs = to_device(media_inputs, device)
            pixel_values = media_inputs["pixel_values"].type(dtype)
            image_embeds = model.visual(
                pixel_values, grid_thw=media_inputs["image_grid_thw"]
            )
            # Add zero-contribution to maintain gradient flow
            inputs_embeds += image_embeds.mean() * 0.0
    else:
        # Process image inputs
        if pixel_values is not None:
            pixel_values = pixel_values.type(dtype)
            image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw)
            
            # Create mask for image token positions
            image_mask = (
                (input_ids == model.config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            
            # Replace image token embeddings with actual image embeddings
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # Process video inputs
        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.type(dtype)
            video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
            
            # Create mask for video token positions
            video_mask = (
                (input_ids == model.config.video_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
            )
            
            # Replace video token embeddings with actual video embeddings
            video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    return {
        "input_ids": input_ids,
        "inputs_embeds": inputs_embeds,
    }


def register_custom_hooks():
    """
    Register custom hooks to Template and Qwen2VLTemplate.
    
    This function should be called explicitly in the training script to patch
    the Swift template classes with custom preprocessing logic.
    
    Example:
        import qwenvl_template_hooks
        qwenvl_template_hooks.register_custom_hooks()
    """
    Template.pre_forward_hook = custom_pre_forward_hook
    Qwen2VLTemplate._post_encode = custom_post_encode


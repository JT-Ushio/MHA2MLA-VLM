import inspect
from typing import Optional, Tuple

import torch
from transformers import Cache
from transformers.utils import logging
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    rotate_half,
    repeat_kv,
    Qwen2_5_VLSdpaAttention,
)


logger = logging.get_logger(__name__)


def create_custom_apply_multimodal_rotary_pos_emb(q_r_indices, k_r_indices):


    def custom_apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
        """
        q: [bs, num_heads, positions, head_dim]
        k: [bs, num_key_value_heads, positions, head_dim]
        cos: [3, bs, positions, head_dim]
        """
        
        frame = inspect.currentframe().f_back
        attention_module = frame.f_locals["self"]
        layer_idx = attention_module.layer_idx
        
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1)
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1)

        # select cos and sin based on the indices
        q_idx = q_r_indices[layer_idx].to(q.device)
        cos_q = cos.repeat(1, 1, q.size(1)).index_select(-1, q_idx)
        sin_q = sin.repeat(1, 1, q.size(1)).index_select(-1, q_idx)
        cos_q = cos_q.reshape(cos_q.size(0), q.size(2), q.size(1), -1).transpose(1, 2)
        sin_q = sin_q.reshape(sin_q.size(0), q.size(2), q.size(1), -1).transpose(1, 2)
        
        k_idx = k_r_indices[layer_idx].to(k.device)
        cos_k = cos.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        sin_k = sin.repeat(1, 1, k.size(1)).index_select(-1, k_idx)
        cos_k = cos_k.reshape(cos_k.size(0), k.size(2), k.size(1), -1).transpose(1, 2)
        sin_k = sin_k.reshape(sin_k.size(0), k.size(2), k.size(1), -1).transpose(1, 2)

        q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
        k_embed = (k * cos_k) + (rotate_half(k) * sin_k)

        return q_embed, k_embed
    return custom_apply_multimodal_rotary_pos_emb

def custom_Qwen2_5_VLSdpaAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "Qwen2_5_VLModel is using Qwen2_5_VLSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        raise ValueError("can not support output_attentions")
        # return super().forward(
        #     hidden_states=hidden_states,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_value=past_key_value,
        #     output_attentions=output_attentions,
        #     use_cache=use_cache,
        #     cache_position=cache_position,
        #     position_embeddings=position_embeddings,
        # )

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    # NOTE: key_states = self.k_proj(hidden_states)
    key_r_states = self.k_r_proj(hidden_states)
    # NOTE: value_states = self.v_proj(hidden_states)
    key_c_states, value_states = self.kv_proj(hidden_states)
    is_gqa2mha2mla = key_c_states.size(-1) + key_r_states.size(-1) != value_states.size(-1)
    n_k_head = self.num_heads if is_gqa2mha2mla else self.num_key_value_heads
    key_r_states = key_r_states.view(bsz, q_len, n_k_head, -1).transpose(1, 2)
    key_c_states = key_c_states.view(bsz, q_len, n_k_head, -1).transpose(1, 2)
    query_r_states = query_states[..., :self.num_heads*key_r_states.size(-1)]
    query_c_states = query_states[..., self.num_heads*key_r_states.size(-1):]
    query_r_states = query_r_states.view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
    query_c_states = query_c_states.view(bsz, q_len, self.num_heads, -1).transpose(1, 2)
    value_states = value_states.view(
        bsz, q_len, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    cos, sin = position_embeddings
    query_r_states, key_r_states = modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb(
        query_r_states, key_r_states, cos, sin, self.rope_scaling["mrope_section"]
    )
    query_states = torch.cat([query_r_states, query_c_states], dim=-1)
    key_states = torch.cat([key_r_states, key_c_states], dim=-1)

    if past_key_value is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "cache_position": cache_position,
        }  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    if not is_gqa2mha2mla:
        key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, -1)

    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value

def mha2mla_qwenvl(q_idx, k_idx):
    Qwen2_5_VLSdpaAttention.forward = custom_Qwen2_5_VLSdpaAttention_forward
    modeling_qwen2_5_vl.apply_multimodal_rotary_pos_emb = create_custom_apply_multimodal_rotary_pos_emb(
        q_idx, k_idx
    )


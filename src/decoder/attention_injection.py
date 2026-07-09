from typing import Any, Optional, Dict
import torch

class ReferenceAttentionProcessor:
    """
    Custom attention processor that caches K and V tensors during keyframe generation
    and injects them during subsequent frame generation to enforce temporal consistency.
    """
    def __init__(self):
        self.reference_kv_cache: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self.is_keyframe = False

    def __call__(
        self,
        attn: Any,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        # Determine if we are doing self-attention (encoder_hidden_states is None)
        # or cross-attention (encoder_hidden_states is not None)
        is_cross_attention = encoder_hidden_states is not None
        
        # Prepare Q, K, V
        query = attn.to_q(hidden_states)
        
        if is_cross_attention:
            key = attn.to_k(encoder_hidden_states)
            value = attn.to_v(encoder_hidden_states)
        else:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
        
        # Generate a unique key for this attention layer using its id or object reference
        layer_id = str(id(attn))
        
        if not is_cross_attention:
            if self.is_keyframe:
                # Cache the K and V tensors for self-attention
                self.reference_kv_cache[layer_id] = (key.detach().clone(), value.detach().clone())
            else:
                # Inject the cached K and V tensors
                if layer_id in self.reference_kv_cache:
                    cached_key, cached_value = self.reference_kv_cache[layer_id]
                    # We inject the cached key and value by concatenating them along the sequence length dimension (dim=1)
                    key = torch.cat([cached_key, key], dim=1)
                    value = torch.cat([cached_value, value], dim=1)

        # Standard attention computation
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(query.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(key.shape[0], -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(value.shape[0], -1, attn.heads, head_dim).transpose(1, 2)

        # scaled_dot_product_attention already scales by 1/sqrt(head_dim) natively
        # Do not scale query here!
        
        hidden_states = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(hidden_states.shape[0], -1, attn.heads * head_dim)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        
        return hidden_states

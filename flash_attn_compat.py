# Flash Attention Compatibility Layer
# Accurate reimplementation of flash_attn for Windows compatibility

import torch
import torch.nn.functional as F
import math
from typing import Optional

# Configuration
USE_HIGH_PRECISION = False  # False: SDPA (fast), True: Manual (precise)
DEBUG_MODE = False  # Set to True for debugging output


def flash_attn_varlen_func_fallback(
    q, k, v, 
    cu_seqlens_q, cu_seqlens_k, 
    max_seqlen_q, max_seqlen_k, 
    dropout_p=0.0, 
    softmax_scale=None, 
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False
):
    """
    Accurate fallback implementation of flash_attn_varlen_func
    
    Args:
        q: [total_q, num_heads, head_dim] - packed query tensor
        k: [total_k, num_heads_k, head_dim] - packed key tensor  
        v: [total_k, num_heads_k, head_dim] - packed value tensor
        cu_seqlens_q: [batch_size + 1] - cumulative sequence lengths for queries
        cu_seqlens_k: [batch_size + 1] - cumulative sequence lengths for keys
        max_seqlen_q: int - maximum query sequence length
        max_seqlen_k: int - maximum key sequence length
        ...
        
    Returns:
        out: [total_q, num_heads, head_dim] - packed output tensor
    """
    
    if DEBUG_MODE:
        print(f"[DEBUG] Input shapes: q={q.shape}, k={k.shape}, v={v.shape}")
        print(f"[DEBUG] cu_seqlens_q={cu_seqlens_q}, cu_seqlens_k={cu_seqlens_k}")
        print(f"[DEBUG] max_seqlen: q={max_seqlen_q}, k={max_seqlen_k}")
    
    # Get dimensions
    total_q = q.size(0)
    num_heads_q = q.size(1)
    head_dim = q.size(2)
    num_heads_k = k.size(1)
    
    batch_size = len(cu_seqlens_q) - 1
    device = q.device
    original_dtype = q.dtype
    
    # Softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    
    # Determine compute dtype
    if USE_HIGH_PRECISION:
        compute_dtype = torch.float32
    else:
        # Keep original dtype for minimal conversion overhead
        compute_dtype = original_dtype
    
    # Convert to compute dtype only if necessary
    if q.dtype != compute_dtype:
        q = q.to(compute_dtype)
        k = k.to(compute_dtype)
        v = v.to(compute_dtype)
    
    # Handle grouped query attention (GQA)
    if num_heads_k != num_heads_q:
        n_rep = num_heads_q // num_heads_k
        # Repeat k and v to match query heads
        k = k.repeat_interleave(n_rep, dim=1)  # [total_k, num_heads_q, head_dim]
        v = v.repeat_interleave(n_rep, dim=1)  # [total_k, num_heads_q, head_dim]
    
    # Process each sequence in the batch
    outputs = []
    
    for i in range(batch_size):
        # Get sequence lengths and slices
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()
        
        q_len = q_end - q_start
        k_len = k_end - k_start
        
        # Extract sequences for this batch element
        # [seq_len, num_heads, head_dim] -> [num_heads, seq_len, head_dim]
        qi = q[q_start:q_end].transpose(0, 1)  # [num_heads_q, q_len, head_dim]
        ki = k[k_start:k_end].transpose(0, 1)  # [num_heads_q, k_len, head_dim]
        vi = v[k_start:k_end].transpose(0, 1)  # [num_heads_q, k_len, head_dim]
        
        if DEBUG_MODE:
            print(f"[DEBUG] Batch {i}: qi={qi.shape}, ki={ki.shape}, vi={vi.shape}")
        
        # Compute attention for this sequence
        if USE_HIGH_PRECISION or causal or dropout_p > 0.0:
            # Manual attention computation
            # [num_heads, q_len, head_dim] x [num_heads, head_dim, k_len] -> [num_heads, q_len, k_len]
            scores = torch.matmul(qi, ki.transpose(-2, -1)) * softmax_scale
            
            # Apply causal mask if needed
            if causal:
                # Flash attention uses bottom-right alignment for causal mask
                # Mask out positions where query position < key position
                if q_len == k_len:
                    # Standard causal mask for same-length sequences
                    causal_mask = torch.triu(
                        torch.ones(q_len, k_len, dtype=torch.bool, device=device),
                        diagonal=1
                    )
                else:
                    # For different lengths, align to bottom-right
                    causal_mask = torch.triu(
                        torch.ones(q_len, k_len, dtype=torch.bool, device=device),
                        diagonal=k_len - q_len + 1
                    )
                scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
            
            # Softmax
            attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)
            
            # Dropout
            if dropout_p > 0.0:
                attn_weights = F.dropout(attn_weights, p=dropout_p)
            
            # Apply attention to values
            out_i = torch.matmul(attn_weights.to(compute_dtype), vi)  # [num_heads, q_len, head_dim]
        else:
            # Use optimized SDPA
            # Add batch dimension: [1, num_heads, seq_len, head_dim]
            qi_batched = qi.unsqueeze(0)
            ki_batched = ki.unsqueeze(0)
            vi_batched = vi.unsqueeze(0)
            
            # Create causal mask if needed
            attn_mask = None
            if causal and q_len > 1:
                # SDPA expects: [batch, num_heads, q_len, k_len] or broadcastable
                # True = masked out, False = kept
                if q_len == k_len:
                    causal_mask = torch.triu(
                        torch.ones(q_len, k_len, dtype=torch.bool, device=device),
                        diagonal=1
                    )
                else:
                    causal_mask = torch.triu(
                        torch.ones(q_len, k_len, dtype=torch.bool, device=device),
                        diagonal=k_len - q_len + 1
                    )
                attn_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, k_len]
            
            try:
                out_i = F.scaled_dot_product_attention(
                    qi_batched, ki_batched, vi_batched,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p if dropout_p > 0.0 else 0.0,
                    is_causal=False,  # We handle causal mask explicitly
                    scale=softmax_scale
                )
                out_i = out_i.squeeze(0)  # Remove batch dimension
            except Exception as e:
                if DEBUG_MODE:
                    print(f"[DEBUG] SDPA failed for batch {i}: {e}, falling back to manual")
                # Fallback to manual
                scores = torch.matmul(qi, ki.transpose(-2, -1)) * softmax_scale
                if causal:
                    if q_len == k_len:
                        causal_mask = torch.triu(
                            torch.ones(q_len, k_len, dtype=torch.bool, device=device),
                            diagonal=1
                        )
                    else:
                        causal_mask = torch.triu(
                            torch.ones(q_len, k_len, dtype=torch.bool, device=device),
                            diagonal=k_len - q_len + 1
                        )
                    scores = scores.masked_fill(causal_mask.unsqueeze(0), float('-inf'))
                attn_weights = F.softmax(scores, dim=-1, dtype=compute_dtype)
                out_i = torch.matmul(attn_weights, vi)
        
        # [num_heads, q_len, head_dim] -> [q_len, num_heads, head_dim]
        out_i = out_i.transpose(0, 1)
        outputs.append(out_i)
    
    # Concatenate all outputs
    output = torch.cat(outputs, dim=0)  # [total_q, num_heads, head_dim]
    
    # Convert back to original dtype if needed
    if output.dtype != original_dtype:
        output = output.to(original_dtype)
    
    if DEBUG_MODE:
        print(f"[DEBUG] Output shape: {output.shape}, dtype: {output.dtype}")
    
    if return_attn_probs:
        # Not implemented for fallback
        return output, None
    else:
        return output

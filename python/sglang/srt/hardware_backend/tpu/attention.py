"""TPU (Sophgo SG2260E) attention backend for SGLang.

The device only produces correct matmuls in bf16, its batched matmul (bmm) is
broken, and ``F.scaled_dot_product_attention`` is unimplemented. So instead of
SDPA this backend reuses :class:`TorchNativeAttnBackend`'s paged KV-cache gather
and per-request extend/decode plumbing, and replaces only the core attention
math with a **per-head loop of contiguous 2D matmuls**:

    scores = (q_i @ k_i.tᵀ)·scale            # bf16 2D matmul, contiguous operands
    scores = scores.float() + causal_mask     # additive float mask, finite -1e9
    p      = softmax(scores).to(bf16)
    out_i  = p @ v_i                          # bf16 2D matmul

This matches a CPU-bf16 reference to <0.5% relative error (verified on hardware).
GQA (Qwen3: more query heads than kv heads) maps each query head to its kv group.
The causal mask is built on CPU (``triu`` and ``-inf`` misbehave on-device) with a
finite large-negative value and moved to the device.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner

# Finite stand-in for -inf in the additive mask: -inf overflows on-device.
_MASK_NEG = -1e9


class TpuAttnBackend(TorchNativeAttnBackend):
    """Per-head 2D-matmul attention for the Sophgo TPU device."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)

    # ------------------------------------------------------------------
    # Core per-request attention math (replaces SDPA)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_additive_mask(
        *,
        q_len: int,
        kv_len: int,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> Optional[torch.Tensor]:
        """Build an additive [q_len, kv_len] float mask on CPU, or None.

        Query token ``i`` sits at absolute position ``query_offset + i`` where
        ``query_offset = kv_len - q_len`` (the extend tokens are the tail of the
        sequence; for decode ``q_len == 1``). A key ``j`` is disallowed (masked to
        ``_MASK_NEG``) when it is in the future (``j > pos``) or, for sliding
        window, older than the window (``j < pos - window``).
        """
        has_window = sliding_window_size is not None and sliding_window_size > -1
        if not causal and not has_window:
            return None

        query_offset = kv_len - q_len
        pos = torch.arange(q_len).unsqueeze(1) + query_offset  # [q_len, 1]
        j = torch.arange(kv_len).unsqueeze(0)  # [1, kv_len]

        disallowed = torch.zeros(q_len, kv_len, dtype=torch.bool)
        if causal:
            disallowed |= j > pos
        if has_window:
            disallowed |= j < pos - sliding_window_size

        mask = torch.zeros(q_len, kv_len, dtype=torch.float32)
        mask.masked_fill_(disallowed, _MASK_NEG)
        return mask

    def _attend_per_req(
        self,
        *,
        q: torch.Tensor,  # [H_q, S_q, D]
        k: torch.Tensor,  # [H_kv, S_kv, D]
        v: torch.Tensor,  # [H_kv, S_kv, D]
        out: torch.Tensor,  # [H_q, S_q, D] (written in place)
        scaling: float,
        causal: bool,
        sliding_window_size: Optional[int],
    ) -> None:
        num_q_heads, q_len, _ = q.shape
        num_kv_heads, kv_len, _ = k.shape
        group_size = num_q_heads // num_kv_heads

        cpu_mask = self._build_additive_mask(
            q_len=q_len,
            kv_len=kv_len,
            causal=causal,
            sliding_window_size=sliding_window_size,
        )
        mask = cpu_mask.to(q.device) if cpu_mask is not None else None

        for h in range(num_q_heads):
            kv_h = h // group_size
            q_i = q[h].contiguous()  # [S_q, D]
            k_i = k[kv_h].contiguous()  # [S_kv, D]
            v_i = v[kv_h].contiguous()  # [S_kv, D]

            # bf16 2D matmul; operands must be contiguous or the result is garbage.
            scores = q_i @ k_i.t().contiguous()  # [S_q, S_kv]
            scores = scores.float() * scaling
            if mask is not None:
                scores = scores + mask
            probs = torch.softmax(scores, dim=-1).to(v_i.dtype)
            out[h] = probs.contiguous() @ v_i  # [S_q, D]

    # ------------------------------------------------------------------
    # Overrides of the SDPA extend/decode drivers
    # ------------------------------------------------------------------

    def _run_sdpa_forward_extend(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
        extend_seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
        assert seq_lens.shape[0] == extend_seq_lens.shape[0]

        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            extend_seq_len_q = int(extend_seq_lens[seq_idx])
            seq_len_kv = int(seq_lens[seq_idx])
            end_q = start_q + extend_seq_len_q
            if encoder_lens is not None:
                if is_cross_attn:
                    start_kv = 0
                    end_kv = int(encoder_lens[seq_idx])
                else:
                    start_kv = int(encoder_lens[seq_idx])
                    end_kv = start_kv + seq_len_kv
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = torch.empty_like(per_req_query)
            self._attend_per_req(
                q=per_req_query,
                k=per_req_key,
                v=per_req_value,
                out=per_req_out,
                scaling=scaling,
                causal=causal,
                sliding_window_size=sliding_window_size,
            )
            output[start_q:end_q, :, :] = per_req_out.movedim(query.dim() - 2, 0)
            start_q, start_kv = end_q, end_kv
        return output

    def _run_sdpa_forward_decode(
        self,
        query: torch.Tensor,
        output: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        req_to_token: torch.Tensor,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor] = None,
        scaling=None,
        enable_gqa=False,
        causal=False,
        is_cross_attn=False,
        sliding_window_size: Optional[int] = None,
    ):
        # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
        query = query.movedim(0, query.dim() - 2)

        start_q, start_kv = 0, 0
        for seq_idx in range(seq_lens.shape[0]):
            seq_len_q = 1
            seq_len_kv = int(seq_lens[seq_idx])
            end_q = start_q + seq_len_q
            if encoder_lens is not None:
                if is_cross_attn:
                    start_kv = 0
                    end_kv = int(encoder_lens[seq_idx])
                else:
                    start_kv = int(encoder_lens[seq_idx])
                    end_kv = start_kv + seq_len_kv
            else:
                start_kv = 0
                end_kv = start_kv + seq_len_kv

            per_req_query = query[:, start_q:end_q, :]

            req_pool_idx = req_pool_indices[seq_idx]
            per_req_tokens = req_to_token[req_pool_idx, start_kv:end_kv]
            per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
            per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

            if not (per_req_query.dtype == per_req_key.dtype == per_req_value.dtype):
                per_req_key = per_req_key.to(per_req_query.dtype)
                per_req_value = per_req_value.to(per_req_query.dtype)

            per_req_out = torch.empty_like(per_req_query)
            self._attend_per_req(
                q=per_req_query,
                k=per_req_key,
                v=per_req_value,
                out=per_req_out,
                scaling=scaling,
                # A decode step attends all cached keys; the driver passes
                # causal=False (every key is in the past).
                causal=causal,
                sliding_window_size=sliding_window_size,
            )
            output[start_q:end_q, :, :] = per_req_out.movedim(query.dim() - 2, 0)
            start_q, start_kv = end_q, end_kv

        return output

    def support_triton(self):
        return False

# Copyright 2026 IBM and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Optional Multi-Token Prediction (MTP) heads for GraniteForDocling.

DeepSeek-V3 / Llama-3 style draft heads used for greedy speculative decoding.
The module is constructed only when ``GraniteForDoclingConfig.use_mtp`` is True;
non-MTP checkpoints keep the original graph and weight layout.

Head ``i`` at position ``t`` reads the previous hidden state and the embedding
of token ``t+1`` and predicts token ``t+2+i``. Heads chain: head ``i+1``
consumes head ``i``'s output. The output projection is the model's tied
``lm_head``.

Weight names match training (``mtp.blocks.{i}.*``). vLLM loads the same
checkpoint when ``use_mtp`` is set; there is no separate draft architecture.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

if TYPE_CHECKING:
    from .configuration_granite_for_docling import GraniteForDoclingConfig


class GraniteForDoclingMTPBlock(nn.Module):
    """One transformer layer for multi-token prediction (DeepSeek-V3 eq. 21).

    Two callable paths that are mathematically identical for matching slots:

    * **Training / standalone** -- pass ``hidden_states`` + ``token_embeddings``
      and get the full causal output. No cache state involved.
    * **Inference with KV cache** -- pass ``past_kv``; returns
      ``(output, new_kv)`` so each speculative round is O(S) rather than O(N).

    Concat order is ``cat[hnorm(hidden), enorm(embed)]``. vLLM uses the same
    order when the heads live on GraniteForDocling.
    """

    def __init__(
        self,
        hidden_size: int,
        nhead: int,
        dim_feedforward: int,
        rms_norm_eps: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.nhead = nhead
        self.head_dim = hidden_size // nhead
        self.input_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.embed_norm = nn.RMSNorm(hidden_size, eps=rms_norm_eps)
        self.proj = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        # Identity-sum init: M_k = [I, I]. Random init of this [d, 2d] matrix
        # made early training diverge (heads confidently wrong, grad_norm tiny).
        with torch.no_grad():
            eye = torch.eye(hidden_size)
            self.proj.weight.copy_(torch.cat([eye, eye], dim=1))
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_embeddings: torch.Tensor,
        past_kv: tuple | None = None,
        return_kv: bool = False,
    ):
        h = self.proj(
            torch.cat(
                [self.input_norm(hidden_states), self.embed_norm(token_embeddings)],
                dim=-1,
            )
        )
        seq_len = h.size(1)
        if not return_kv and past_kv is None:
            if seq_len > 1:
                mask = nn.Transformer.generate_square_subsequent_mask(
                    seq_len, device=h.device, dtype=h.dtype
                )
                return self.transformer_layer(h, src_mask=mask, is_causal=True)
            return self.transformer_layer(h)
        return self._forward_with_kv(h, past_kv)

    def _forward_with_kv(self, h: torch.Tensor, past_kv: tuple | None):
        layer = self.transformer_layer
        B, S, H = h.shape

        x_norm = layer.norm1(h)
        qkv = nn.functional.linear(
            x_norm, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias
        )
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, S, self.nhead, self.head_dim)
        k = k.view(B, S, self.nhead, self.head_dim)
        v = v.view(B, S, self.nhead, self.head_dim)

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(1)
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)
        new_kv = (k, v)

        q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if past_kv is None:
            attn_out = nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=(S > 1)
            )
        elif S == 1:
            attn_out = nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=False
            )
        else:
            total = past_len + S
            mask = torch.zeros((S, total), device=h.device, dtype=h.dtype)
            idx = torch.arange(S, device=h.device)
            allowed = past_len + idx + 1
            col = torch.arange(total, device=h.device).unsqueeze(0)
            mask = mask.masked_fill(col >= allowed.unsqueeze(1), float("-inf"))
            attn_out = nn.functional.scaled_dot_product_attention(
                q_t, k_t, v_t, attn_mask=mask, is_causal=False
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, H)
        attn_out = nn.functional.linear(
            attn_out, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias
        )

        h = h + attn_out
        x_norm2 = layer.norm2(h)
        ff = layer.linear2(layer.activation(layer.linear1(x_norm2)))
        return h + ff, new_kv


class GraniteForDoclingMTP(nn.Module):
    """Optional K-head MTP module attached to ``GraniteForDoclingForConditionalGeneration``.

    Parameter prefix: ``mtp.blocks.{i}.{embed_norm,input_norm,proj,transformer_layer.*}``.
    """

    def __init__(self, config: GraniteForDoclingConfig):
        super().__init__()
        text = config.text_config
        hidden_size = text.hidden_size
        nhead = int(getattr(config, "mtp_num_heads_attn", None) or text.num_attention_heads)
        dim_ff = int(getattr(config, "mtp_ffn_dim", None) or text.intermediate_size)
        rms_eps = getattr(text, "rms_norm_eps", 1e-5)
        dropout = float(getattr(config, "mtp_dropout", 0.0) or 0.0)
        n_heads = int(getattr(config, "mtp_num_heads", 0) or 0)
        if n_heads <= 0:
            raise ValueError("GraniteForDoclingMTP requires mtp_num_heads > 0")
        if hidden_size % nhead:
            raise ValueError(
                f"hidden_size={hidden_size} is not divisible by mtp nhead={nhead}"
            )
        self.num_heads = n_heads
        self.hidden_size = hidden_size
        self.loss_weight = float(getattr(config, "mtp_weight", 0.3) or 0.0)
        self.loss_chunk_size = int(getattr(config, "mtp_loss_chunk_size", 1024) or 0)
        self.blocks = nn.ModuleList(
            [
                GraniteForDoclingMTPBlock(
                    hidden_size, nhead, dim_ff, rms_eps, dropout=dropout
                )
                for _ in range(n_heads)
            ]
        )

    def compute_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.Tensor,
        embed_fn: nn.Module,
        lm_head: nn.Module,
        logits_scaling: float = 1.0,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, list[float | None]]:
        """Teacher-forced MTP auxiliary loss.

        ``labels`` are already next-token labels (``labels[t] == input_ids[t+1]``,
        ``-100`` ignored). Head ``i`` at slot ``q`` reads ``embed(labels[q+i])``
        and predicts ``labels[q+i+1]``.

        Returns ``(loss, detached_mean, per_head_means)``.
        """
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0, -100)

        n_heads = len(self.blocks)
        head_ce = [hidden_states.new_zeros(()) for _ in range(n_heads)]
        head_n = [0] * n_heads
        chunk = self.loss_chunk_size

        for row, beg, end in _mtp_segments(labels):
            seg_h = hidden_states[row : row + 1, beg:end, :]
            seg_y = labels[row : row + 1, beg:end]
            seg_len = seg_h.size(1)
            prev_hidden = seg_h

            for i, block in enumerate(self.blocks):
                span = seg_len - (i + 2)
                if span <= 0:
                    break
                prev_trimmed = prev_hidden[:, :span, :]
                interv_targets = seg_y[:, i : i + span]
                interv_embeds = embed_fn(interv_targets.clamp(min=0))
                interv_embeds = interv_embeds.masked_fill(
                    (interv_targets == -100).unsqueeze(-1), 0.0
                )
                block_hidden = block(prev_trimmed, interv_embeds)
                ce_sum, n_valid = _mtp_head_ce(
                    block_hidden,
                    seg_y[:, i + 1 : i + 1 + span],
                    lm_head,
                    logits_scaling,
                    chunk,
                    training=self.training,
                )
                head_ce[i] = head_ce[i] + ce_sum
                head_n[i] += n_valid
                prev_hidden = block_hidden

        total = None
        used = 0
        anchor = None
        for i in range(n_heads):
            if head_n[i] == 0:
                z = sum(p.sum() for p in self.blocks[i].parameters()) * 0.0
                anchor = z if anchor is None else anchor + z
                continue
            mean_i = head_ce[i] / head_n[i]
            if not torch.isfinite(mean_i):
                z = sum(p.sum() for p in self.blocks[i].parameters()) * 0.0
                anchor = z if anchor is None else anchor + z
                continue
            total = mean_i if total is None else total + mean_i
            used += 1

        if total is None or used == 0:
            if anchor is None:
                return hidden_states.new_zeros(()), 0.0, [None] * n_heads
            return anchor, 0.0, [None] * n_heads
        mtp_loss = total / used
        if anchor is not None:
            mtp_loss = mtp_loss + anchor
        per_head = [
            (float((head_ce[i] / head_n[i]).detach()) if head_n[i] else None)
            for i in range(n_heads)
        ]
        return mtp_loss, float(mtp_loss.detach()), per_head


def _mtp_segments(labels: torch.Tensor) -> list[tuple[int, int, int]]:
    """Yield ``(row, start, end)`` contiguous spans. Each batch row is one span."""
    rows, cols = labels.shape
    return [(r, 0, cols) for r in range(rows)]


def _mtp_head_ce(
    block_hidden: torch.Tensor,
    labels: torch.Tensor,
    lm_head: nn.Module,
    logits_scaling: float,
    chunk_size: int,
    training: bool,
) -> tuple[torch.Tensor, int]:
    flat_h = block_hidden.reshape(-1, block_hidden.size(-1))
    flat_y = labels.reshape(-1)
    n_valid = int((flat_y != -100).sum().item())
    if n_valid == 0:
        return flat_h.new_zeros(()), 0

    def _chunk_ce(h_chunk, y_chunk):
        lg = lm_head(h_chunk)
        if logits_scaling != 1.0:
            lg = lg / logits_scaling
        return F.cross_entropy(lg, y_chunk, ignore_index=-100, reduction="sum")

    if chunk_size <= 0:
        return _chunk_ce(flat_h, flat_y), n_valid

    total = flat_h.new_zeros(())
    for beg in range(0, flat_h.size(0), chunk_size):
        h_c = flat_h[beg : beg + chunk_size]
        y_c = flat_y[beg : beg + chunk_size]
        if training and torch.is_grad_enabled():
            total = total + torch.utils.checkpoint.checkpoint(
                _chunk_ce, h_c, y_c, use_reentrant=False
            )
        else:
            total = total + _chunk_ce(h_c, y_c)
    return total, n_valid


def maybe_build_mtp(config: GraniteForDoclingConfig) -> GraniteForDoclingMTP | None:
    """Return an MTP module when ``use_mtp`` is set, else ``None``."""
    if not bool(getattr(config, "use_mtp", False)):
        return None
    n_heads = int(getattr(config, "mtp_num_heads", 0) or 0)
    if n_heads <= 0:
        return None
    return GraniteForDoclingMTP(config)


def generate_speculative(
    model,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None = None,
    pixel_values: torch.Tensor | None = None,
    pixel_attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 32,
    eos_token_id: int | None = None,
    **model_kwargs,
) -> torch.LongTensor:
    """Greedy speculative decoding using the attached MTP heads.

    Every emitted token is the token the backbone itself would have produced
    (up to bf16 near-tie numerics). Requires ``model.mtp`` to be built.
    """
    mtp: GraniteForDoclingMTP = model.mtp
    if mtp is None:
        raise RuntimeError("generate_speculative requires config.use_mtp=True")

    text_model = model.model.text_model
    lm_head = model.lm_head
    embed_fn = text_model.get_input_embeddings()
    scaling = getattr(model.config.text_config, "logits_scaling", 1.0) or 1.0
    n_blocks = len(mtp.blocks)

    def _scale(x):
        return x if scaling == 1.0 else x / scaling

    def _pick(logits):
        return torch.argmax(logits, dim=-1, keepdim=True)

    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        use_cache=True,
        **model_kwargs,
    )
    hidden = outputs.last_hidden_state
    past_kv = outputs.past_key_values
    batch_size = hidden.size(0)
    current_seq_len = hidden.size(1)
    mtp_hidden_states = hidden

    next_token_id = _pick(_scale(lm_head(hidden[:, -1, :])))
    generated_ids = [next_token_id]
    if attention_mask is not None:
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (batch_size, 1),
                    device=attention_mask.device,
                    dtype=attention_mask.dtype,
                ),
            ],
            dim=1,
        )
    if eos_token_id is not None and torch.any(next_token_id == eos_token_id):
        return torch.cat(generated_ids, dim=1)

    tokens_generated = 1
    prefill_len = mtp_hidden_states.size(1)
    committed_response_tokens: list[torch.Tensor] = []
    mtp_caches: list = [None] * n_blocks
    block_outputs_so_far: list = [None] * n_blocks

    while tokens_generated < max_new_tokens:
        outputs = text_model(
            input_ids=next_token_id,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
        )
        hidden = outputs.last_hidden_state
        past_kv = outputs.past_key_values
        current_seq_len += 1
        mtp_hidden_states = torch.cat([mtp_hidden_states, hidden], dim=1)
        committed_response_tokens.append(next_token_id)
        _assert_mtp_alignment(mtp_hidden_states, committed_response_tokens, prefill_len)

        main_token = _pick(_scale(lm_head(hidden[:, -1, :])))
        generated_ids.append(main_token)
        tokens_generated += 1
        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, 1),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
        if eos_token_id is not None and torch.any(main_token == eos_token_id):
            break
        if tokens_generated >= max_new_tokens:
            break

        T_full = mtp_hidden_states.size(1)
        B, H = mtp_hidden_states.size(0), mtp_hidden_states.size(2)

        def intervening_for_chunk(shift, chunk_start, chunk_end, main_token=None, drafts=()):
            L = chunk_end - chunk_start
            embeds = torch.zeros(
                (B, L, H), device=mtp_hidden_states.device, dtype=mtp_hidden_states.dtype
            )
            crt = (
                torch.cat(committed_response_tokens, dim=1)
                if committed_response_tokens
                else None
            )
            n_committed = crt.size(1) if crt is not None else 0
            for local_idx in range(L):
                target_pos = chunk_start + local_idx + shift
                if target_pos < prefill_len:
                    continue
                horizon = prefill_len + n_committed
                if target_pos < horizon:
                    embeds[:, local_idx, :] = embed_fn(crt[:, target_pos - prefill_len])
                    continue
                offset = target_pos - horizon
                tok = None
                if offset == 0:
                    tok = main_token
                elif 1 <= offset <= len(drafts):
                    tok = drafts[offset - 1]
                if tok is not None:
                    embeds[:, local_idx, :] = embed_fn(tok).squeeze(1)
            return embeds

        draft_tokens = []
        for i, block in enumerate(mtp.blocks):
            old_size = mtp_caches[i][0].size(1) if mtp_caches[i] is not None else 0
            source = mtp_hidden_states if i == 0 else block_outputs_so_far[i - 1]
            if source.size(1) != T_full:
                raise RuntimeError(
                    f"MTP head {i} source trace has {source.size(1)} slots, expected {T_full}"
                )
            prev_h_chunk = source[:, old_size:T_full, :]
            interv = intervening_for_chunk(
                i + 1, old_size, T_full, main_token=main_token, drafts=draft_tokens
            )
            chunk_out, new_kv = block(
                prev_h_chunk, interv, past_kv=mtp_caches[i], return_kv=True
            )
            mtp_caches[i] = new_kv
            block_outputs_so_far[i] = (
                chunk_out
                if block_outputs_so_far[i] is None
                else torch.cat([block_outputs_so_far[i], chunk_out], dim=1)
            )
            draft_tokens.append(
                torch.argmax(_scale(lm_head(chunk_out[:, -1, :])), dim=-1, keepdim=True)
            )

        k = len(draft_tokens)
        cache_len_before_verify = current_seq_len
        verify_input = torch.cat([main_token] + draft_tokens[:-1], dim=1)
        verify_attn = attention_mask
        if verify_attn is not None and k > 1:
            verify_attn = torch.cat(
                [
                    verify_attn,
                    torch.ones(
                        (batch_size, k - 1),
                        device=verify_attn.device,
                        dtype=verify_attn.dtype,
                    ),
                ],
                dim=1,
            )
        outputs = text_model(
            input_ids=verify_input,
            attention_mask=verify_attn,
            past_key_values=past_kv,
            use_cache=True,
        )
        verify_hidden = outputs.last_hidden_state
        past_kv = outputs.past_key_values
        all_verify_logits = _scale(lm_head(verify_hidden))

        first_reject = k
        accepted_extra = []
        hit_stop = False
        for i in range(k):
            if tokens_generated >= max_new_tokens:
                break
            verified = _pick(all_verify_logits[:, i, :])
            if bool(torch.all(verified == draft_tokens[i])):
                accepted_extra.append(draft_tokens[i])
                tokens_generated += 1
                if eos_token_id is not None and torch.any(draft_tokens[i] == eos_token_id):
                    hit_stop = True
                    break
            else:
                accepted_extra.append(verified)
                tokens_generated += 1
                first_reject = i
                if eos_token_id is not None and torch.any(verified == eos_token_id):
                    hit_stop = True
                break

        n_correct = min(first_reject + 1, k)
        if n_correct < k:
            target_len = cache_len_before_verify + n_correct
            if hasattr(past_kv, "crop"):
                past_kv.crop(target_len)
            current_seq_len = target_len
        else:
            current_seq_len += k

        for bi in range(1, n_blocks):
            if first_reject <= bi - 1:
                if mtp_caches[bi] is not None:
                    k_c, v_c = mtp_caches[bi]
                    if k_c.size(1) > 0:
                        mtp_caches[bi] = (k_c[:, :-1, :, :], v_c[:, :-1, :, :])
                if (
                    block_outputs_so_far[bi] is not None
                    and block_outputs_so_far[bi].size(1) > 0
                ):
                    block_outputs_so_far[bi] = block_outputs_so_far[bi][:, :-1, :]

        mtp_hidden_states = torch.cat(
            [mtp_hidden_states, verify_hidden[:, :n_correct, :]], dim=1
        )
        committed_response_tokens.append(main_token)
        for j in range(min(first_reject, k - 1)):
            committed_response_tokens.append(draft_tokens[j])
        _assert_mtp_alignment(mtp_hidden_states, committed_response_tokens, prefill_len)

        next_token_id = accepted_extra[-1] if accepted_extra else main_token
        generated_ids.extend(accepted_extra)
        if attention_mask is not None and accepted_extra:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (batch_size, len(accepted_extra)),
                        device=attention_mask.device,
                        dtype=attention_mask.dtype,
                    ),
                ],
                dim=1,
            )
        if hit_stop:
            break

    if not generated_ids:
        return torch.empty((batch_size, 0), dtype=torch.long, device=input_ids.device)
    return torch.cat(generated_ids, dim=1)


def _assert_mtp_alignment(mtp_hidden_states, committed_response_tokens, prefill_len):
    n_hidden = mtp_hidden_states.size(1) - prefill_len
    n_tokens = len(committed_response_tokens)
    if n_hidden != n_tokens:
        raise RuntimeError(
            f"MTP position desync: hidden trace holds {n_hidden} response positions "
            f"but {n_tokens} tokens are committed."
        )


__all__ = [
    "GraniteForDoclingMTP",
    "GraniteForDoclingMTPBlock",
    "generate_speculative",
    "maybe_build_mtp",
]

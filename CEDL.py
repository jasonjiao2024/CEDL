"""CEDL: contextual encoding, expansion, directed retrieval and linkage.

The deployed model combines CEDLBackbone with ProbabilityReadout. Inputs are
unpadded token IDs; position t predicts token t+1. Every call is an independent
request. PyTorch is the only runtime dependency.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

__all__ = ["CEDLConfig", "CEDLBackbone", "ProbabilityReadout", "CEDL",
           "configure_execution", "load_checkpoint", "save_checkpoint"]
__version__ = "1.0.0"
_FORMAT = "cedl-1"
_PROJECTION_ROWS = 256


@dataclass(frozen=True)
class CEDLConfig:
    vocab_size: int = 50257
    d_model: int = 448
    n_heads: int = 7
    c_layers: int = 6
    ffn_dim: int = 2048
    expansion: int = 4
    max_context: int = 2048
    dropout: float = 0.0
    activation_checkpointing: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "d_model", "n_heads", "c_layers", "ffn_dim",
                     "expansion", "max_context"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.vocab_size < 2 or self.d_model < 4 or self.d_model % self.n_heads:
            raise ValueError("vocab_size >= 2 and d_model divisible by n_heads are required")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if type(self.activation_checkpointing) is not bool:
            raise ValueError("activation_checkpointing must be boolean")


def configure_execution(cpu_threads: int = 1) -> None:
    """Explicitly set the FP32 execution policy; importing this file changes none."""
    if type(cpu_threads) is not int or cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(cpu_threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _linear(x: Tensor, weight: Tensor, bias: Tensor | None = None,
            *, canonical: bool = True) -> Tensor:
    if not canonical or x.ndim < 2 or x.numel() == 0:
        return F.linear(x, weight, bias)
    flat = x.reshape(-1, x.shape[-1])
    parts = []
    for start in range(0, len(flat), _PROJECTION_ROWS):
        part = flat[start:start + _PROJECTION_ROWS]
        count = len(part)
        if count < _PROJECTION_ROWS:
            part = torch.cat((part, part.new_zeros(_PROJECTION_ROWS - count, part.shape[-1])))
        parts.append(F.linear(part, weight, bias)[:count])
    return torch.cat(parts).reshape(*x.shape[:-1], weight.shape[0])


class _Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return _linear(x, self.weight, self.bias, canonical=not self.training)


def _retention_scan(q: Tensor, k: Tensor, v: Tensor, decay: Tensor,
                    write: Tensor) -> Tensor:
    b, t, h, d = q.shape
    size = min(32, t)
    n = (t + size - 1) // size
    pad = n * size - t

    def padded(x, fill=0):
        return torch.cat((x, x.new_full((b, pad, *x.shape[2:]), fill)), 1) if pad else x

    def blocks(x):
        return x.reshape(b, n, size, h, -1).permute(0, 3, 1, 2, 4)

    qb, kb, vb = (blocks(padded(x)) for x in (q, k, v))
    a = padded(decay, 1).reshape(b, n, size, h).permute(0, 3, 1, 2).log()
    w = padded(write).reshape(b, n, size, h).permute(0, 3, 1, 2)
    lower = torch.ones(size, size, dtype=torch.bool, device=q.device).tril()
    strict = lower & ~torch.eye(size, dtype=torch.bool, device=q.device)
    segments = a.unsqueeze(-1).expand(*a.shape, size).masked_fill(~strict, 0).cumsum(-2)
    segments = segments.masked_fill(~lower, -torch.inf).exp()
    prefix = a.cumsum(-1).exp()
    values = torch.cat((vb, torch.ones_like(vb[..., :1])), -1)
    local = ((qb @ kb.transpose(-1, -2)) * segments * w.unsqueeze(-2)) @ values
    end_weight = segments[..., -1, :] * w
    update = kb.transpose(-1, -2) @ (values * end_weight.unsqueeze(-1))
    multiplier = prefix[..., -1, None, None]
    offset = 1
    while offset < n:
        update = torch.cat((update[:, :, :offset], update[:, :, offset:] +
                            multiplier[:, :, offset:] * update[:, :, :-offset]), 2)
        multiplier = torch.cat((multiplier[:, :, :offset], multiplier[:, :, offset:] *
                                multiplier[:, :, :-offset]), 2)
        offset *= 2
    initial = q.new_zeros(b, h, d, v.shape[-1] + 1)
    ends = update + multiplier * initial.unsqueeze(2)
    starts = torch.cat((initial.unsqueeze(2), ends[:, :, :-1]), 2)
    projected = local + (qb @ starts) * prefix.unsqueeze(-1)
    output = projected[..., :-1] / projected[..., -1:].clamp_min(1e-4)
    return output.permute(0, 2, 3, 1, 4).reshape(b, n * size, h, v.shape[-1])[:, :t]


class _Retention(nn.Module):
    def __init__(self, c: CEDLConfig):
        super().__init__()
        d, h = c.d_model, c.n_heads
        self.n_heads, self.d_head = h, d // h
        for name in ("w_q", "w_k", "w_v", "w_out", "out_gate_proj"):
            setattr(self, name, _Linear(d, d))
        gammas = torch.linspace(.85, .995, h)
        self.gamma_log = nn.Parameter(torch.log(gammas / (1 - gammas)))
        self.gn = nn.GroupNorm(h, d)
        self.decay_projection = _Linear(d, h)
        self.write_projection = _Linear(d, h)
        self.dropout = nn.Dropout(c.dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, t, width = x.shape
        h, d = self.n_heads, self.d_head
        q = (F.elu(self.w_q(x), alpha=1.) + 1.).view(b, t, h, d)
        k = (F.elu(self.w_k(x), alpha=1.) + 1.).view(b, t, h, d)
        v = self.w_v(x).view(b, t, h, d)
        decay_logits = self.gamma_log.to(x.dtype).view(1, 1, h) + self.decay_projection(x).to(x.dtype)
        decay = decay_logits.sigmoid().clamp(1e-4, 1 - 1e-4)
        write = self.write_projection(x).to(x.dtype).sigmoid()
        with torch.autocast(device_type=x.device.type, enabled=False):
            output = _retention_scan(*(z.float() for z in (q, k, v, decay, write)))
        grouped = output.to(x.dtype).reshape(b, t, h, d)
        mean = grouped.mean(-1, keepdim=True)
        variance = grouped.var(-1, keepdim=True, unbiased=False)
        normalized = (grouped - mean) * torch.rsqrt(variance + self.gn.eps)
        normalized = normalized * self.gn.weight.view(1, 1, h, d)
        normalized = normalized + self.gn.bias.view(1, 1, h, d)
        gate = 2. * self.out_gate_proj(x).sigmoid()
        return self.w_out(self.dropout(normalized.reshape(b, t, width) * gate))


class _CLayer(nn.Module):
    def __init__(self, c: CEDLConfig):
        super().__init__()
        d = c.d_model
        self.ln1 = nn.LayerNorm(d, elementwise_affine=False)
        self.retention = _Retention(c)
        self.ln2 = nn.LayerNorm(d, elementwise_affine=False)
        self.ffn = nn.Sequential(_Linear(d, c.ffn_dim), nn.GELU(), nn.Dropout(c.dropout),
                                 _Linear(c.ffn_dim, d), nn.Dropout(c.dropout))
        self.neuro_proj = nn.Sequential(_Linear(d, d // 4), nn.GELU(), _Linear(d // 4, 4 * d))

    def forward(self, x: Tensor, feedback: Tensor | None = None) -> Tensor:
        if feedback is None:
            x = x + self.retention(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        s1, b1, s2, b2 = self.neuro_proj(feedback).chunk(4, -1)
        s1, b1 = s1.clamp(-.5, .5), b1.clamp(-.5, .5)
        s2, b2 = s2.clamp(-.5, .5), b2.clamp(-.5, .5)
        x = x + self.retention((1 + s1) * self.ln1(x) + b1)
        return x + self.ffn((1 + s2) * self.ln2(x) + b2)


class _ContextualEncoding(nn.Module):
    def __init__(self, c: CEDLConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)
        self.drop = nn.Dropout(c.dropout)
        self.layers = nn.ModuleList(_CLayer(c) for _ in range(c.c_layers))
        self.ln = nn.LayerNorm(c.d_model)
        self.activation_checkpointing = c.activation_checkpointing

    def forward(self, ids: Tensor, feedback: Tensor | None = None) -> Tensor:
        hidden = self.drop(self.tok_emb(ids))
        for layer in self.layers:
            if self.training and self.activation_checkpointing:
                hidden = checkpoint(layer, hidden, feedback, use_reentrant=False)
            else:
                hidden = layer(hidden, feedback)
        return self.ln(hidden)


class _Expansion(nn.Module):
    def __init__(self, c: CEDLConfig):
        super().__init__()
        expanded = c.d_model * c.expansion
        self.expand = _Linear(c.d_model, expanded)
        self.contract = _Linear(expanded, c.d_model)
        self.ln = nn.LayerNorm(c.d_model)
        self.register_buffer("neuron_freq", torch.zeros(expanded))
        self.register_buffer("softmax_temperature", torch.tensor(1.))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        code = F.gelu(self.expand(x))
        if self.training:
            with torch.no_grad():
                threshold = code.detach().abs().mean(-1, keepdim=True)
                frequency = code.detach().abs().ge(threshold).float().mean((0, 1))
                self.neuron_freq.mul_(.99).add_(frequency, alpha=1 - .99)
        return self.ln(self.contract(code) + x), code

    @staticmethod
    def separation_loss(context: Tensor, code: Tensor) -> Tensor:
        length = context.shape[1]
        if length > 128:
            indices = torch.linspace(0, length - 1, 128, device=context.device).long()
            context, code = context.index_select(1, indices), code.index_select(1, indices)
            length = 128
        a = F.normalize(context.float(), dim=-1, eps=1e-6)
        b = F.normalize(code.float(), dim=-1, eps=1e-6)
        similarity = F.relu(torch.bmm(a, a.transpose(1, 2))).detach()
        overlap = F.relu(torch.bmm(b, b.transpose(1, 2)))
        off_diagonal = 1. - torch.eye(length, device=context.device).unsqueeze(0)
        return (similarity * overlap * off_diagonal).sum() / (len(context) * length * max(length - 1, 1))


def _gather(values: Tensor, indices: Tensor) -> Tensor:
    return values.gather(1, indices.unsqueeze(-1).expand(*indices.shape, values.shape[-1]))


def _segment_sum(values: Tensor, groups: Tensor) -> Tensor:
    total, offset = values, 1
    while offset < groups.shape[1]:
        same = groups[:, offset:].eq(groups[:, :-offset]).unsqueeze(-1)
        prior = torch.cat((torch.zeros_like(total[:, :offset]),
                           torch.where(same, total[:, :-offset], torch.zeros_like(total[:, :-offset]))), 1)
        total = total + prior
        offset *= 2
    return total


class _ReadRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(_Linear(8, 16), nn.Tanh(), _Linear(16, 1))

    def forward(self, features: Tensor) -> Tensor:
        return self.layers(features.to(self.layers[0].weight.dtype)).sigmoid().squeeze(-1)


def _successor_read(ids: Tensor, records: Tensor, router: _ReadRouter):
    b, t = ids.shape
    dtype = torch.float64 if records.dtype == torch.float64 else torch.float32
    values = records.to(dtype)
    shifted = torch.cat((values[:, 1:], torch.zeros_like(values[:, :1])), 1)
    pos = torch.arange(t, device=ids.device).expand(b, -1)
    payload = torch.cat((shifted, torch.ones_like(pos, dtype=dtype).unsqueeze(-1),
                         pos.to(dtype).unsqueeze(-1)), -1)
    totals_list, counts_list = [], []
    codes, radix = ids.long(), int(ids.max()) + 2
    best_source = torch.zeros_like(ids)
    for lag in range(3):
        if lag:
            past = torch.cat((torch.full_like(ids[:, :min(lag, t)], -1), ids[:, :max(t - lag, 0)]), 1)
            codes = groups * radix + past + 1
        order = codes.argsort(dim=1, stable=True)
        sorted_codes = codes.gather(1, order)
        starts = torch.cat((torch.ones_like(sorted_codes[:, :1], dtype=torch.bool),
                            sorted_codes[:, 1:].ne(sorted_codes[:, :-1])), 1)
        grouped = starts.long().cumsum(1) - 1
        groups = torch.zeros_like(grouped).scatter(1, order, grouped)
        sorted_pos = pos.gather(1, order)
        keys = (grouped * (t + 1) + sorted_pos).contiguous()
        upper, lower = pos - 2, torch.full_like(pos, lag - 1)
        hi = torch.searchsorted(keys, (groups * (t + 1) + upper).contiguous(), right=True) - 1
        lo = torch.searchsorted(keys, (groups * (t + 1) + lower).contiguous(), right=True) - 1
        prefix = _segment_sum(_gather(payload, order), grouped)

        def at(rank):
            safe = rank.clamp(0, t - 1)
            valid = rank.ge(0) & grouped.gather(1, safe).eq(groups)
            return _gather(prefix, safe) * valid.unsqueeze(-1).to(dtype)

        total = (at(hi) - at(lo)) * upper.gt(lower).unsqueeze(-1).to(dtype)
        count = total[..., -2].round().long()
        best_source = torch.where(count.gt(0), sorted_pos.gather(1, hi.clamp(0, t - 1)), best_source)
        if lag == 0:
            first = sorted_pos.gather(1, (lo + 1).clamp(0, t - 1))
        totals_list.append(total)
        counts_list.append(count)
    coefficients = [math.exp(-2)] + [math.exp(lag - 2) * (-math.expm1(-1)) for lag in (1, 2)]
    total = sum(c * x for c, x in zip(coefficients, totals_list))
    counts = torch.stack(counts_list, -1)
    denominator = total[..., -2].clamp_min(torch.finfo(dtype).tiny)
    present = counts[..., 0].gt(0)
    soft = total[..., :-2] / denominator.unsqueeze(-1)
    soft = torch.where(present.unsqueeze(-1), soft, torch.zeros_like(soft))
    sharp = _gather(values, (best_source + 1).clamp_max(t - 1)) * present.unsqueeze(-1)
    lengths = torch.arange(3, device=ids.device)
    longest = torch.where(counts.gt(0), lengths, torch.zeros_like(lengths)).max(-1).values
    categories = counts - torch.cat((counts[..., 1:], torch.zeros_like(counts[..., :1])), -1)
    log_weight = lengths.to(dtype) - 2
    expected_log = (categories.to(dtype) * log_weight.exp() * log_weight).sum(-1) / denominator
    entropy = (denominator.log() - expected_log).clamp_min(0)
    top_mass = (longest.to(dtype) - 2).exp() / denominator
    zero = torch.zeros_like(denominator)
    entropy = torch.where(present, entropy, zero)
    top_mass = torch.where(present, top_mass, zero)
    normalized_entropy = entropy / counts[..., 0].to(dtype).clamp_min(2).log()
    n = counts[..., 0].float()
    best = counts.float().gather(-1, longest.unsqueeze(-1)).squeeze(-1).clamp_min(1)
    positions = torch.arange(t, device=ids.device).float()
    features = torch.stack((n.log1p() / 4, best.log1p() / 4, longest.float() / 2,
        (n / best).log1p() / 4, top_mass, normalized_entropy,
        (positions - best_source.float()) / positions.clamp_min(1), best.gt(1).float()), -1)
    gate = router(features).to(soft.dtype) * present.to(soft.dtype)
    mixed = soft + gate.unsqueeze(-1) * (sharp - soft)
    index = torch.where(present, first + 1, torch.zeros_like(first))
    return mixed.to(records.dtype), present, index, counts[..., 0]


def _shift(x: Tensor, amount: int) -> Tensor:
    if amount >= x.shape[1]:
        return torch.zeros_like(x)
    return torch.cat((x.new_zeros(len(x), amount, x.shape[-1]), x[:, :-amount]), 1)


class _RelationalMemory(nn.Module):
    def __init__(self, c: CEDLConfig):
        super().__init__()
        d = c.d_model
        self.record_address = _Linear(2 * d, max(24, d // 3), bias=False)
        self.query_address = _Linear(2 * d, max(24, d // 3), bias=False)
        self.record_value = _Linear(d, d, bias=False)
        self.read_output = _Linear(d, d, bias=False)
        self.validity = nn.Sequential(_Linear(d, max(16, d // 4)), nn.GELU(), _Linear(max(16, d // 4), 1))
        self.read_norm = nn.LayerNorm(d)
        self.log_similarity_scale = nn.Parameter(torch.tensor(math.log(8.)))
        self.raw_version_strength = nn.Parameter(torch.tensor(math.log(math.expm1(1.))))
        self.value_mix_logit = nn.Parameter(torch.zeros(d))
        self.token_value_adapter = _Linear(d, d, bias=False)
        self.exact_mix_gate = _Linear(3 * d + 2, 1)
        self.adaptive_router = _ReadRouter()

    def _fallback(self, x: Tensor):
        b, t, _ = x.shape
        m1, m2, m3 = (_shift(x, lag) for lag in (1, 2, 3))
        keys = F.normalize(self.record_address(torch.cat((m3, m2), -1)), dim=-1, eps=1e-6)
        queries = F.normalize(self.query_address(torch.cat((m2, m1), -1)), dim=-1, eps=1e-6)
        records = x + self.record_value(x)
        scores = torch.einsum("btd,bsd->bts", queries, keys) * self.log_similarity_scale.exp().clamp(1, 32)
        scores = scores + self.validity(m1).squeeze(-1).unsqueeze(1)
        positions = torch.arange(t, device=x.device)
        target, source = positions.view(t, 1), positions.view(1, t)
        causal = (source.lt(target) & source.ge(1)).view(1, t, t).expand(b, -1, -1)
        version = source.to(scores.dtype) / target.clamp_min(1).to(scores.dtype)
        scores = scores + F.softplus(self.raw_version_strength) * version.unsqueeze(0)
        pages = (t + 15) // 16
        eligible = scores.masked_fill(~causal, -1e4)
        if pages * 16 > t:
            eligible = F.pad(eligible, (0, pages * 16 - t), value=-1e4)
        selected = eligible.view(b, t, pages, 16).amax(-1).topk(min(2, pages), dim=-1).indices
        active = causal & selected.unsqueeze(-1).eq((positions // 16).view(1, 1, 1, t)).any(2)
        weights = F.softmax(scores.masked_fill(~active, -1e4), -1) * active.to(scores.dtype)
        den = weights.sum(-1, keepdim=True)
        weights = torch.where(den.gt(0), weights / den.clamp_min(1e-8), torch.zeros_like(weights))
        retrieved = torch.einsum("bts,bsd->btd", weights, records)
        output = self.read_norm(x + retrieved + self.read_output(retrieved))
        return output, weights.detach()

    def forward(self, x: Tensor, ids: Tensor, embeddings: Tensor):
        fallback_output, weights = self._fallback(x)
        records = x + self.record_value(x)
        # Address weights are stop-gradient on the mixed branch of this model.
        retrieved = torch.einsum("bts,bsd->btd", weights.to(records.dtype), records)
        token_records = F.normalize(embeddings.float(), dim=-1, eps=1e-6)
        token_records = token_records.mul(math.sqrt(x.shape[-1])).to(x.dtype)
        token_records = token_records + self.token_value_adapter(token_records)
        value_mix = self.value_mix_logit.sigmoid().view(1, 1, -1).to(x.dtype)
        exact_records = value_mix * (x + self.record_value(x)) + (1 - value_mix) * token_records
        exact, present, index, count = _successor_read(ids, exact_records, self.adaptive_router)
        pos = torch.arange(x.shape[1], device=x.device)
        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        scale = torch.log1p(pos.clamp_min(1).to(dtype))
        distance = torch.log1p((pos.unsqueeze(0) - index).clamp_min(0).to(dtype)) / scale
        log_count = torch.log1p(count.to(dtype)) / scale
        inputs = torch.cat((x, exact, retrieved, distance.to(x.dtype).unsqueeze(-1),
                            log_count.to(x.dtype).unsqueeze(-1)), -1)
        mixture = self.exact_mix_gate(inputs).sigmoid()
        mixture = mixture * present.unsqueeze(-1).to(mixture.dtype)
        mixed = mixture.to(exact.dtype) * exact + (1 - mixture.to(retrieved.dtype)) * retrieved
        proposed = self.read_norm(x + mixed + self.read_output(mixed))
        output = torch.where(present.unsqueeze(-1), proposed, fallback_output)
        return output, (present, count, mixture.squeeze(-1).detach())


class _DirectedRetrieval(nn.Module):
    def __init__(self, c: CEDLConfig):
        super().__init__()
        d = c.d_model
        self.specialist_scale = nn.Parameter(torch.zeros(d))
        self.ln_loop2 = nn.LayerNorm(d)
        self.relational_memory = _RelationalMemory(c)
        self.specialist_gate = _Linear(3 * d, d)

    def forward(self, x: Tensor, ids: Tensor, embeddings: Tensor):
        baseline = self.ln_loop2(x)
        snapshot, evidence = self.relational_memory(x, ids, embeddings)
        difference = snapshot - baseline
        gate = self.specialist_gate(torch.cat((baseline, snapshot, difference.abs()), -1)).sigmoid()
        residual = self.specialist_scale.tanh().view(1, 1, -1) * gate * difference
        return baseline + residual, evidence


class _FeedbackAttention(nn.MultiheadAttention):
    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        t, b, d, h = query.shape[1], len(query), self.embed_dim, self.num_heads
        future = torch.ones(t, t, device=query.device, dtype=torch.bool).triu(1)
        if self.training:
            return super().forward(query, context, context, attn_mask=future, need_weights=False)[0]
        # Packed Q / KV projections preserve the time-major projection order.
        q0, k0 = query.transpose(0, 1), context.transpose(0, 1)
        wq, wkv = self.in_proj_weight.split((d, 2 * d))
        bq, bkv = self.in_proj_bias.split((d, 2 * d))
        q = _linear(q0, wq, bq)
        kv = _linear(k0, wkv, bkv).unflatten(-1, (2, d)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()
        q = q.view(t, b * h, d // h).transpose(0, 1).view(b, h, t, d // h)
        k, v = (z.view(t, b * h, d // h).transpose(0, 1).view(b, h, t, d // h) for z in kv)
        mask = torch.zeros_like(future, dtype=query.dtype).masked_fill(future, -torch.inf)[None, None]
        attended = F.scaled_dot_product_attention(q, k, v, mask, 0., False)
        attended = attended.permute(2, 0, 1, 3).contiguous().view(b * t, d)
        return _linear(attended, self.out_proj.weight, self.out_proj.bias).view(t, b, d).transpose(0, 1)


class _Linkage(nn.Module):
    def __init__(self, c: CEDLConfig):
        super().__init__()
        d = c.d_model
        self.mem_head = _Linear(d, c.vocab_size)
        self.per_head = _Linear(d, c.vocab_size, bias=False)
        self.gate_net = nn.Sequential(_Linear(2 * d, 128), nn.GELU(), _Linear(128, 1), nn.Sigmoid())
        self.loop1_attn = _FeedbackAttention(d, c.n_heads, dropout=c.dropout, batch_first=True)
        self.loop1_gate = nn.Sequential(_Linear(2 * d, d), nn.Sigmoid())
        self.ln_loop1 = nn.LayerNorm(d)
        self.read_feature_gate = _Linear(4, 1, bias=False)

    def feedback(self, memory: Tensor, context: Tensor) -> Tensor:
        fused = torch.full_like(memory, .5) * memory + (1 - torch.full_like(memory, .5)) * context
        attended = self.loop1_attn(fused, context)
        gate = self.loop1_gate(torch.cat((fused, attended), -1))
        return self.ln_loop1(fused + gate * attended)

    def compare(self, memory: Tensor, context: Tensor, evidence):
        present, count, mixture = evidence
        positions = torch.arange(count.shape[1], device=count.device).float()
        disagreement = (memory.detach().float() - context.detach().float()).norm(dim=-1)
        disagreement = disagreement / (memory.detach().float().norm(dim=-1) +
                                        context.detach().float().norm(dim=-1) + 1e-6)
        features = torch.stack((present.float(), mixture.float(),
            count.float().log1p() / positions.clamp_min(1).log1p(), disagreement), -1).detach()
        logit = torch.cat((memory, context), -1)
        for layer in list(self.gate_net.children())[:-1]:
            logit = layer(logit)
        gate = torch.sigmoid(logit + self.read_feature_gate(features.to(memory.dtype)))
        return gate * memory + (1 - gate) * context, gate


class CEDLBackbone(nn.Module):
    """Native CEDL decoder; forward returns (logits, unscaled auxiliary loss).

    `score_mask` selects projection positions, not padding or attention entries.
    Training labels must already be shifted: labels[:, t] is the next token.
    """
    def __init__(self, config: CEDLConfig = CEDLConfig()):
        super().__init__()
        self.config = config
        self.register_buffer("feedback_alpha", torch.tensor(1.))
        self.c_stage = _ContextualEncoding(config)
        self.e_stage = _Expansion(config)
        self.d_stage = _DirectedRetrieval(config)
        self.l_stage = _Linkage(config)
        self.l_stage.mem_head.weight = self.c_stage.tok_emb.weight
        self.l_stage.per_head.weight = self.c_stage.tok_emb.weight
        self._initialize()

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, 0., .02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
        for layer in self.c_stage.layers:
            nn.init.zeros_(layer.neuro_proj[-1].weight)
            nn.init.zeros_(layer.neuro_proj[-1].bias)
            for projection in (layer.retention.decay_projection, layer.retention.write_projection):
                nn.init.zeros_(projection.weight)
                nn.init.zeros_(projection.bias)
            nn.init.constant_(layer.retention.write_projection.bias, 2.)
        memory = self.d_stage.relational_memory
        with torch.no_grad():
            memory.query_address.weight.copy_(memory.record_address.weight)
        for module in (memory.record_value, memory.read_output, memory.token_value_adapter,
                       self.l_stage.read_feature_gate):
            nn.init.zeros_(module.weight)
        nn.init.zeros_(memory.adaptive_router.layers[-1].weight)
        nn.init.constant_(memory.adaptive_router.layers[-1].bias, -4.)

    def _validate(self, ids: Tensor, mask: Tensor | None = None):
        if ids.ndim != 2 or ids.dtype not in (torch.int32, torch.int64) or len(ids) < 1:
            raise ValueError("input_ids must be integer [batch, time]")
        if not 1 <= ids.shape[1] <= self.config.max_context:
            raise ValueError("input length is outside configured max_context")
        if ids.device != self.c_stage.tok_emb.weight.device:
            raise ValueError("input_ids and backbone must be on the same device")
        if ids.device.type not in ("cpu", "cuda"):
            raise ValueError("this execution policy supports CPU and CUDA")
        if bool(((ids < 0) | (ids >= self.config.vocab_size)).any()):
            raise ValueError("token ID is outside the vocabulary")
        if mask is not None and (mask.shape != ids.shape or mask.dtype != torch.bool or
                                 mask.device != ids.device or not bool(mask.any())):
            raise ValueError("score_mask must be a nonempty boolean selection aligned with input_ids")

    def _sweep(self, ids: Tensor, feedback: Tensor | None = None):
        context = self.c_stage(ids, feedback)
        expanded, code = self.e_stage(context)
        memory, evidence = self.d_stage(expanded, ids, self.c_stage.tok_emb(ids))
        return context, code, memory, evidence

    def forward(self, input_ids: Tensor, score_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        self._validate(input_ids, score_mask)
        reference, first_code, first_memory, _ = self._sweep(input_ids)
        feedback = self.l_stage.feedback(first_memory, reference)
        _, _, memory, evidence = self._sweep(input_ids, self.feedback_alpha * feedback)
        hidden, gate = self.l_stage.compare(memory, reference, evidence)
        auxiliary = memory.new_zeros(())
        if self.training:
            if input_ids.shape[1] < 2:
                raise ValueError("training auxiliary losses require at least two tokens")
            separation = self.e_stage.separation_loss(reference, first_code)
            variance = first_memory.reshape(-1, first_memory.shape[-1]).var(dim=0)
            regularization = F.relu(.1 - variance).mean()
            predictive = F.smooth_l1_loss(F.normalize(first_memory[:, :-1], dim=-1),
                                          F.normalize(reference[:, 1:].detach(), dim=-1))
            auxiliary = .05 * separation + .10 * regularization + .05 * predictive
        if score_mask is not None:
            hidden, gate = hidden[score_mask], gate[score_mask]
        logits = _linear(hidden, self.l_stage.mem_head.weight, canonical=not self.training)
        return logits + (gate * self.l_stage.mem_head.bias).to(logits.dtype), auxiliary

    def loss(self, input_ids: Tensor, labels: Tensor) -> Tensor:
        """Native language objective; labels are aligned next-token IDs or -100."""
        if labels.shape != input_ids.shape or labels.dtype != torch.long or labels.device != input_ids.device:
            raise ValueError("labels must be aligned int64 next-token targets")
        if not bool(labels.ne(-100).any()):
            raise ValueError("at least one target is required")
        logits, auxiliary = self(input_ids)
        return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten()) + .2 * auxiliary


class ProbabilityReadout(nn.Module):
    """Request-reset RNN controlling a bounded-weight lexical probability mixture."""
    def __init__(self):
        super().__init__()
        self.register_buffer("theta", torch.zeros(2, 16))
        self.register_buffer("mean", torch.zeros(16))
        self.register_buffer("std", torch.ones(16))
        self.core = nn.RNN(16, 20, batch_first=True)
        self.out = nn.Linear(20, 2)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, features: Tensor) -> Tensor:
        if features.device.type != "cpu" or features.dtype != torch.float32:
            raise ValueError("readout features must be CPU FP32")
        x = ((features - self.mean) / self.std).clamp(-10, 10)
        hidden, _ = self.core(x)
        return F.linear(features, self.theta) + self.out(hidden)


def _cache_sources(ids: Tensor):
    t = ids.shape[1]
    query = torch.arange(t, device=ids.device)
    source = query[None]
    eligible = ids[0, None].eq(ids[0, :, None]) & (source <= query[:, None] - 2)
    live, lengths = eligible.clone(), torch.zeros_like(eligible, dtype=torch.long)
    for lag in (1, 2):
        live = (live & (source >= lag) & (query[:, None] >= lag) &
                ids[0, (source - lag).clamp_min(0)].eq(ids[0, (query - lag).clamp_min(0)][:, None]))
        lengths += live.long()
    present, longest = eligible.any(-1), lengths.max(-1).values
    best = eligible & lengths.eq(longest[:, None])
    latest = torch.where(best, source, -1).max(-1).values.clamp_min(0)
    token = ids[0, (source + 1).clamp_max(t - 1)].expand(t, -1)
    vocabulary, inverse = ids.unique(sorted=True, return_inverse=True)
    inverse = inverse.reshape_as(ids)[0, (source + 1).clamp_max(t - 1)].expand(t, -1)
    hist = torch.zeros(t, len(vocabulary) * 3, dtype=torch.int64, device=ids.device)
    hist.scatter_add_(-1, inverse * 3 + lengths, eligible.long())
    hist = hist.reshape(t, len(vocabulary), 3)
    totals = hist.sum(1)
    mass = torch.zeros(t, len(vocabulary), dtype=torch.float64, device=ids.device)
    denominator = torch.zeros(t, 1, dtype=torch.float64, device=ids.device)
    coeff = [math.exp(level - 2) for level in range(3)]
    for level, c in enumerate(coeff):
        mass = mass + hist[:, :, level].double() * c
        denominator = denominator + totals[:, level, None].double() * c
    den = denominator.clamp_min(1e-30)
    cache = (mass / den).float()
    weights = (torch.tensor(coeff, dtype=torch.float64, device=ids.device)[lengths] * eligible / den).float()
    mode = torch.where(present, vocabulary[mass.argmax(-1)], 0)
    best_hist = hist.gather(-1, longest[:, None, None].expand(-1, len(vocabulary), 1)).squeeze(-1)
    best_count = best.sum(-1)
    best_uniform = (best_hist.double() / best_count[:, None].clamp_min(1)).float()
    latest_index = inverse.gather(-1, latest[:, None])
    sharp = torch.zeros(t, len(vocabulary), device=ids.device).scatter(-1, latest_index, present.float()[:, None])
    return dict(cache=cache, sharp=sharp, weights=weights, mode=mode, token=token, vocabulary=vocabulary,
        present=present, count=eligible.sum(-1), best_count=best_count, longest=longest,
        latest_token=token.gather(-1, latest[:, None]).squeeze(-1), best_uniform=best_uniform,
        latest_age=(query - latest).float() / query.clamp_min(1))


def _cache_features(logp: Tensor, s: dict) -> Tensor:
    w, mode = s["weights"], s["mode"]
    mode_logp = logp.gather(-1, mode[:, None]).squeeze(-1)
    latest_logp = logp.gather(-1, s["latest_token"][:, None]).squeeze(-1)
    count, best_count, prob = s["count"].float(), s["best_count"].float(), logp.exp()
    entropy = -(w * w.clamp_min(1e-30).log()).sum(-1) / count.clamp_min(2).log()
    return torch.stack((torch.ones_like(count), count.log1p() / 4, s["longest"].float() / 2,
        w.max(-1).values, entropy, prob.max(-1).values,
        -(prob * prob.clamp_min(1e-30).log()).sum(-1) / math.log(prob.shape[-1]), mode_logp.exp(),
        best_count.log1p() / 4, ((count + 1) / (best_count + 1)).log() / 4,
        s["latest_age"] * s["present"], s["best_uniform"].max(-1).values,
        latest_logp.exp() * s["present"], mode.eq(s["latest_token"]).float() * s["present"],
        mode_logp.clamp_min(-30) / 10, latest_logp.clamp_min(-30) / 10 * s["present"]), -1)


def _mixture_logp(base: Tensor, cache: Tensor, gate: Tensor) -> Tensor:
    tiny = torch.finfo(base.dtype).tiny
    extra = (gate.clamp_min(tiny).log() + cache.clamp_min(tiny).log()).masked_fill(
        gate.eq(0) | cache.eq(0), -torch.inf)
    return torch.logaddexp(torch.log1p(-gate) + base, extra)


class CEDL(nn.Module):
    """Deployed CEDL with a CPU FP32 readout and CPU/CUDA FP32 backbone.

    `.to(device)` moves the backbone; the small readout remains on CPU. Dense
    forward returns normalized log probabilities. `score` returns per-target
    NLL and greedy IDs on CPU. Both require eval mode and one unpadded request.
    """
    def __init__(self, config: CEDLConfig = CEDLConfig()):
        super().__init__()
        self.config = config
        self.backbone = CEDLBackbone(config)
        self.readout = ProbabilityReadout()

    def _apply(self, fn, recurse=True):
        probe = fn(torch.empty(0, dtype=torch.float32))
        if probe.dtype != torch.float32 or probe.device.type not in ("cpu", "cuda"):
            raise ValueError("deployed CEDL requires FP32 weights on CPU or CUDA")
        self.backbone._apply(fn, recurse=recurse)
        return self

    @property
    def device(self):
        return self.backbone.c_stage.tok_emb.weight.device

    def _prepare(self, ids: Tensor, mask: Tensor | None):
        self.backbone._validate(ids, mask)
        if len(ids) != 1:
            raise ValueError("deployed inference takes one unpadded request per call")
        if self.training or self.backbone.training or self.readout.training:
            raise RuntimeError("call model.eval() for deployed inference; train through model.backbone")
        if self.backbone.c_stage.tok_emb.weight.dtype != torch.float32:
            raise ValueError("deployed inference requires FP32 backbone weights")
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        with torch.autocast(device_type=ids.device.type, enabled=False):
            logits = self.backbone(ids, torch.ones_like(ids, dtype=torch.bool))[0].float()
            logp = logits.log_softmax(-1)
            source = _cache_sources(ids)
            features = _cache_features(logp, source).reshape(1, ids.shape[1], 16).cpu()
        with torch.autocast(device_type="cpu", enabled=False):
            controls = self.readout(features)[mask.cpu()]
        if not torch.isfinite(logp).all() or not torch.isfinite(controls).all():
            raise FloatingPointError("nonfinite probabilities or readout controls")
        positions = mask[0].nonzero().flatten()
        selected = {k: (v if k == "vocabulary" else v[positions]) for k, v in source.items()}
        return logits, logp, selected, controls, mask

    @torch.no_grad()
    def forward(self, input_ids: Tensor, score_mask: Tensor | None = None) -> Tensor:
        """Log probabilities [time, vocabulary], or [selected, vocabulary]."""
        _, logp, source, controls, mask = self._prepare(input_ids, score_mask)
        controls = controls.to(logp.device)
        gate = .5 * controls[:, 0].sigmoid() * source["present"]
        mix = controls[:, 1].sigmoid()
        cache = (1 - mix[:, None]) * source["cache"] + mix[:, None] * source["sharp"]
        selected = logp[mask.flatten()]
        indices = source["vocabulary"].expand(len(selected), -1)
        base = torch.log1p(-gate[:, None]) + selected
        updated = _mixture_logp(selected.gather(-1, indices), cache, gate[:, None])
        return base.scatter(-1, indices, updated)

    log_probabilities = forward

    @torch.no_grad()
    def score(self, input_ids: Tensor, targets: Tensor, score_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return per-target NLL and predictions; targets never enter gate features."""
        if targets.ndim != 1 or targets.dtype != torch.long or bool(((targets < 0) | (targets >= self.config.vocab_size)).any()):
            raise ValueError("targets must be one-dimensional int64 vocabulary IDs")
        expected = input_ids.numel() if score_mask is None else int(score_mask.sum())
        if len(targets) != expected:
            raise ValueError("one target is required for every selected position")
        logits, _, source, controls, mask = self._prepare(input_ids, score_mask)
        lp = logits[mask.flatten()].log_softmax(-1)
        gold = targets.to(lp.device)
        vocab = source["vocabulary"]
        match = vocab[None].eq(gold[:, None])
        base_gold = lp.gather(-1, gold[:, None]).squeeze(-1).cpu()
        cache_gold = (source["cache"] * match).sum(-1).cpu()
        sharp_gold = (source["sharp"] * match).sum(-1).cpu()
        top = lp.argmax(-1)
        choices = torch.cat((top[:, None], torch.where(source["weights"] > 0, source["token"], top[:, None])), -1).sort(-1).values
        ci = torch.searchsorted(vocab, choices).clamp_max(len(vocab) - 1)
        valid = vocab[ci].eq(choices)
        choice_base = lp.gather(-1, choices).cpu()
        choice_cache = (source["cache"].gather(-1, ci) * valid).cpu()
        choice_sharp = (source["sharp"].gather(-1, ci) * valid).cpu()
        gate = .5 * controls[:, 0].sigmoid() * source["present"].cpu()
        mix = controls[:, 1].sigmoid()
        nll = -_mixture_logp(base_gold, (1 - mix) * cache_gold + mix * sharp_gold, gate)
        scores = _mixture_logp(choice_base, (1 - mix[:, None]) * choice_cache + mix[:, None] * choice_sharp, gate[:, None])
        prediction = choices.cpu().gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        return nll, prediction

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 32, eos_token_id: int | None = None) -> Tensor:
        """Greedy generation by full-prefix recomputation; no persistent KV cache."""
        if type(max_new_tokens) is not int or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        if eos_token_id is not None and (type(eos_token_id) is not int or not 0 <= eos_token_id < self.config.vocab_size):
            raise ValueError("invalid eos_token_id")
        self.backbone._validate(input_ids)
        if len(input_ids) != 1:
            raise ValueError("generation takes one request")
        if input_ids.shape[1] + max_new_tokens > self.config.max_context:
            raise ValueError("prompt plus generation exceeds max_context; input is never silently truncated")
        result = input_ids.clone()
        for _ in range(max_new_tokens):
            mask = torch.zeros_like(result, dtype=torch.bool)
            mask[:, -1] = True
            _, token = self.score(result, torch.zeros(1, dtype=torch.long), mask)
            result = torch.cat((result, token.to(result.device).view(1, 1)), -1)
            if eos_token_id is not None and int(token[0]) == eos_token_id:
                break
        return result


def _strict_state(module: nn.Module, state: Mapping[str, Tensor]) -> None:
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint state must be a tensor mapping")
    expected = module.state_dict()
    if set(state) != set(expected):
        raise ValueError(f"checkpoint keys differ: missing={sorted(set(expected) - set(state))}, extra={sorted(set(state) - set(expected))}")
    for key, tensor in state.items():
        if not isinstance(tensor, Tensor) or tensor.shape != expected[key].shape or tensor.dtype != expected[key].dtype:
            raise ValueError(f"checkpoint shape or dtype mismatch: {key}")
        if tensor.is_floating_point() and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"nonfinite checkpoint tensor: {key}")
    prefix = "backbone." if isinstance(module, CEDL) else ""
    if isinstance(module, (CEDL, CEDLBackbone)):
        embedding = state[prefix + "c_stage.tok_emb.weight"]
        for key in ("l_stage.mem_head.weight", "l_stage.per_head.weight"):
            if not torch.equal(embedding, state[prefix + key]):
                raise ValueError("tied embedding/projection copies disagree")
        if float(state[prefix + "feedback_alpha"]) != 1.:
            raise ValueError("checkpoint changes the fixed feedback strength")
    std_key = "readout.std" if isinstance(module, CEDL) else "std"
    if std_key in state and bool((state[std_key] < .1).any()):
        raise ValueError("readout feature scales must be at least 0.1")
    module.load_state_dict(state, strict=True)


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu",
                    *, expected_sha256: str | None = None) -> CEDL:
    """Load a standalone tensor-only checkpoint and return the model in eval mode."""
    if expected_sha256 is not None and _sha256(path) != expected_sha256:
        raise ValueError("checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError("not a standalone CEDL checkpoint; use the export command")
    if set(payload) != {"format", "config", "state_dict", "provenance"}:
        raise ValueError("unexpected checkpoint fields")
    model = CEDL(CEDLConfig(**payload["config"]))
    _strict_state(model, payload["state_dict"])
    return model.to(device).eval()


def save_checkpoint(model: CEDL, path: str | Path, *, provenance: dict | None = None) -> None:
    """Atomically create a checkpoint without overwriting an existing file."""
    if not isinstance(model, CEDL):
        raise TypeError("expected a deployed CEDL model")
    provenance = {} if provenance is None else provenance
    json.dumps(provenance, allow_nan=False)
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    _strict_state(model, state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".cedl-", suffix=".pt", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            torch.save(dict(format=_FORMAT, config=asdict(model.config), state_dict=state,
                            provenance=provenance), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="combine matching backbone and RNN readout weights")
    export.add_argument("--backbone", type=Path, required=True)
    export.add_argument("--readout", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--config", type=Path, help="JSON CEDLConfig; defaults to the 448-wide model")
    export.add_argument("--backbone-sha256", required=True)
    export.add_argument("--readout-sha256", required=True)
    run = commands.add_parser("generate", help="greedy continuation of comma-separated token IDs")
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--sha256", required=True)
    run.add_argument("--ids", required=True)
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--max-new-tokens", type=int, default=32)
    run.add_argument("--eos-token-id", type=int)
    args = parser.parse_args()
    configure_execution()
    if args.command == "export":
        for path, expected in ((args.backbone, args.backbone_sha256), (args.readout, args.readout_sha256)):
            if _sha256(path) != expected:
                raise ValueError(f"SHA256 mismatch: {path.name}")
        config = CEDLConfig(**json.loads(args.config.read_text())) if args.config else CEDLConfig()
        model = CEDL(config)
        backbone = torch.load(args.backbone, map_location="cpu", weights_only=True)
        readout = torch.load(args.readout, map_location="cpu", weights_only=True)
        if not isinstance(backbone, dict) or "model_state" not in backbone:
            raise ValueError("backbone checkpoint must contain model_state")
        if not isinstance(readout, dict) or "state" not in readout or readout.get("kind", "rnn") != "rnn":
            raise ValueError("readout checkpoint must contain an RNN state")
        _strict_state(model.backbone, backbone["model_state"])
        _strict_state(model.readout, readout["state"])
        save_checkpoint(model, args.output, provenance={"backbone_sha256": args.backbone_sha256,
            "readout_sha256": args.readout_sha256, "backbone_contract": backbone.get("contract_sha256"),
            "readout_contract": readout.get("contract_sha256")})
        print(json.dumps({"checkpoint": str(args.output), "sha256": _sha256(args.output)}))
    else:
        model = load_checkpoint(args.checkpoint, args.device, expected_sha256=args.sha256)
        ids = torch.tensor([[int(x.strip()) for x in args.ids.split(",")]], device=args.device)
        result = model.generate(ids, args.max_new_tokens, args.eos_token_id)
        print(json.dumps({"token_ids": result[0].tolist()}))


if __name__ == "__main__":
    _main()

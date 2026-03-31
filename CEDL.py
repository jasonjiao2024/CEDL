#!/usr/bin/env python3
"""
CEDL: A Hippocampal-Inspired Architecture for Language Modeling
===============================================================

Reference implementation for the CEDL architecture at ~100M parameter scale,
trained on WikiText-103 and evaluated via perplexity and LAMBADA accuracy.

Architecture:
    C-Stage (EC)  — Grid-cell periodic attention with per-layer AdaLN
    E-Stage (DG)  — Expansion + top-k with AHSD and CSR pattern separation
    D-Stage (CA3) — Dual memory: attractor refinement + 256-slot memory bank
    L-Stage (CA1) — Two-channel comparator with learned fusion gate
    Feedback      — Loop 1 (L→C via AdaLN), Loop 2 (D→E additive)
    NeuromodGate  — Output-level gain modulation

Baselines (parameter-matched):
    Transformer, Transformer-XL, RetNet, Mamba, LSTM

Usage:
    python cedl_release.py --model CEDL --batch-size 4 --max-steps 30000
    python cedl_release.py --model all --batch-size 8 --max-steps 30000
    python cedl_release.py --model all --eval-only

Paper: "CEDL: A Hippocampal-Inspired Architecture for Advancing LLMs"
Author: Dian Jiao (University of Pennsylvania)
License: MIT
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Hyperparameters for 100M-scale training."""
    # Data
    dataset: str = "wikitext103"           # "wikitext103" or "c4"
    vocab_size: int = 50257                # GPT-2 tokenizer
    max_seq: int = 1024                    # context length
    # Training
    batch_size: int = 4                    # per device
    grad_accum: int = 4                    # effective = batch_size * n_devices * grad_accum
    lr: float = 3e-4
    min_lr: float = 1e-5
    warmup_steps: int = 1000
    max_steps: int = 30_000                # ~30 epochs of WikiText-103
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    bfloat16: bool = True
    # Eval
    eval_interval: int = 1000
    eval_steps: int = 200
    # Checkpointing
    save_interval: int = 5000
    save_dir: str = "checkpoints_100m"
    # Hardware
    tpu: bool = False
    seed: int = 42


# ---------------------------------------------------------------------------
# Model Configurations (all ~100M params)
# ---------------------------------------------------------------------------

@dataclass
class CEDLConfig:
    d_model: int = 640
    n_heads: int = 10
    c_layers: int = 8              # up from 6 (params recovered from tied mem_head)
    ffn_dim: int = 2816            # sized to hit ~100M total params
    e_expand: int = 4
    e_sparsity: float = 0.10      # down from 0.20 (closer to biological DG ~5%)
    d_refine: int = 3              # attractor settling steps
    d_slots: int = 256             # explicit memory bank slots
    dropout: float = 0.1
    # Feedback
    n_feedback_iters: int = 1
    feedback_decay: float = 0.5    # geometric decay per iteration
    feedback_warmup_start: int = 2000
    feedback_warmup_end: int = 5000
    # Sparsity annealing
    sparsity_final: float = 0.05   # anneal to 5% after 80% of training
    sparsity_anneal_frac: float = 0.80


@dataclass
class TransformerConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 9
    ffn_dim: int = 3072
    dropout: float = 0.1


@dataclass
class TransformerXLConfig:
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 9
    ffn_dim: int = 3072
    mem_len: int = 256       # reduced from 512 to fit 22GB VRAM
    dropout: float = 0.1


@dataclass
class RetNetConfig:
    d_model: int = 640
    n_heads: int = 10
    n_layers: int = 12
    ffn_dim: int = 2880      # tuned for ~102M params
    dropout: float = 0.1


@dataclass
class MambaConfig:
    d_model: int = 768
    n_layers: int = 18       # ~109M params (slightly over due to 2x expand)
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2          # inherent to Mamba architecture; 2x expansion
    dropout: float = 0.0     # Mamba paper uses no dropout (standard practice)


@dataclass
class LSTMConfig:
    d_model: int = 640      # embedding dim
    hidden_size: int = 1760  # tuned for ~100M params
    n_layers: int = 3
    dropout: float = 0.1


# ============================================================================
# Multi-Scale Retention (shared by CEDL C-stage and RetNet baseline)
# ============================================================================

class MultiScaleRetention(nn.Module):
    """Parallel multi-scale retention (RetNet-style)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.out_gate_proj = nn.Linear(d_model, d_model)
        self.gn = nn.GroupNorm(n_heads, d_model)

        # Learnable decay rates (multi-scale). Range [0.80, 0.99] is wider
        # than the original RetNet paper's 1-2^(-5-n) (~[0.969, 0.999]) to
        # give heads a broader range of temporal scales.
        gammas = torch.linspace(0.80, 0.99, n_heads)
        self.gamma_log = nn.Parameter(
            torch.log(gammas / (1.0 - gammas))  # inverse sigmoid
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        K = self.n_heads
        dh = self.d_head

        Q = self.w_q(x).view(B, T, K, dh).transpose(1, 2)   # [B, K, T, dh]
        Kmat = self.w_k(x).view(B, T, K, dh).transpose(1, 2)
        V = self.w_v(x).view(B, T, K, dh).transpose(1, 2)

        # Build decay matrix: D[k,i,j] = gamma_k^(i-j) for i>=j, else 0
        gamma = torch.sigmoid(self.gamma_log)             # [K]
        log_gamma = torch.log(gamma)                      # [K]
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        # diff[i,j] = i - j; positive in lower triangle where i > j
        diff = (positions.unsqueeze(1) - positions.unsqueeze(0)).clamp(min=0)
        D = torch.exp(log_gamma.view(K, 1, 1) * diff.unsqueeze(0))  # [K, T, T]
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        D = D * causal.unsqueeze(0)                       # [K, T, T]

        # Retention
        retention = (Q @ Kmat.transpose(-1, -2)) * D.unsqueeze(0)  # [B, K, T, T]
        row_sum = retention.sum(dim=-1, keepdim=True).abs().clamp(min=1.0)
        retention = retention / row_sum
        out = retention @ V                               # [B, K, T, dh]

        # Reshape + GroupNorm + output gate
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.gn(out.transpose(1, 2)).transpose(1, 2)
        gate = F.silu(self.out_gate_proj(x))
        out = self.w_out(self.dropout(out * gate))
        return out


class MultiScalePeriodicRetention(nn.Module):
    """Grid-cell-inspired periodic retention for CEDL C-stage.

    Replaces RetNet's exponential decay D[i,j] = γ^(i-j) with a damped
    periodic kernel D[i,j] = γ^(i-j) · cos(2π(i-j)/λ_k + φ_k), inspired
    by entorhinal cortex grid cells that fire at regular spatial intervals
    at multiple scales. No existing sequence model uses periodic attention
    weights — this is the key novelty of the C-stage.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert d_model % n_heads == 0

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)
        self.out_gate_proj = nn.Linear(d_model, d_model)
        self.gn = nn.GroupNorm(n_heads, d_model)

        # Decay envelope (same as RetNet)
        gammas = torch.linspace(0.85, 0.995, n_heads)
        self.gamma_log = nn.Parameter(
            torch.log(gammas / (1.0 - gammas)))

        # Grid cell periodic parameters (NOVEL)
        # Learnable period lengths spanning 4 to 256 tokens
        self.log_lambda = nn.Parameter(
            torch.linspace(math.log(4.0), math.log(256.0), n_heads))
        # Learnable phase offsets
        self.phi = nn.Parameter(torch.zeros(n_heads))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        K = self.n_heads
        dh = self.d_head

        Q = self.w_q(x).view(B, T, K, dh).transpose(1, 2)
        Kmat = self.w_k(x).view(B, T, K, dh).transpose(1, 2)
        V = self.w_v(x).view(B, T, K, dh).transpose(1, 2)

        # Decay envelope
        gamma = torch.sigmoid(self.gamma_log)
        log_gamma = torch.log(gamma)

        # Grid cell periods and phases
        lam = torch.exp(self.log_lambda)
        positions = torch.arange(T, device=x.device, dtype=x.dtype)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # [T, T]
        diff_pos = diff.clamp(min=0)

        # Damped periodic kernel: γ^(i-j) · cos(2π(i-j)/λ + φ)
        decay = torch.exp(log_gamma.view(K, 1, 1) * diff_pos.unsqueeze(0))
        periodic = torch.cos(
            2 * math.pi * diff.unsqueeze(0) / lam.view(K, 1, 1)
            + self.phi.view(K, 1, 1))
        D = decay * periodic
        causal = torch.tril(torch.ones(T, T, device=x.device, dtype=x.dtype))
        D = D * causal.unsqueeze(0)

        # Retention with periodic kernel
        retention = (Q @ Kmat.transpose(-1, -2)) * D.unsqueeze(0)
        # abs().sum() handles negative D values from cosine
        row_sum = retention.abs().sum(dim=-1, keepdim=True).clamp(min=1.0)
        retention = retention / row_sum
        out = retention @ V

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.gn(out.transpose(1, 2)).transpose(1, 2)
        gate = F.silu(self.out_gate_proj(x))
        out = self.w_out(self.dropout(out * gate))
        return out


class PeriodicRetentionLayer(nn.Module):
    """Pre-norm periodic retention layer with hierarchical neuromodulatory gating.

    Each layer's LayerNorm is modulated by feedback from the L-stage
    mismatch detector via learned scale+shift (Adaptive LayerNorm).
    This distributes neuromodulation across ALL C-stage layers, matching
    how cholinergic projections from the medial septum innervate every
    level of the hippocampal circuit — not just a single output stage.

    Zero-initialized so modulation starts as identity (no warmup needed).
    The model learns to use feedback gradually through training.
    """

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.retention = MultiScalePeriodicRetention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        # Hierarchical neuromodulatory gating: feedback -> (scale1, shift1, scale2, shift2)
        # Bottlenecked to save params: d -> d//4 -> 4*d
        bottleneck = d_model // 4
        self.neuro_proj = nn.Sequential(
            nn.Linear(d_model, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 4 * d_model),
        )
        # Zero-init: starts as identity (scale=1, shift=0) — no warmup needed
        nn.init.zeros_(self.neuro_proj[2].weight)
        nn.init.zeros_(self.neuro_proj[2].bias)

    def forward(self, x: torch.Tensor,
                feedback: Optional[torch.Tensor] = None) -> torch.Tensor:
        if feedback is not None:
            # Neuromodulatory gating: feedback modulates LayerNorm at each sublayer
            params = self.neuro_proj(feedback)
            s1, b1, s2, b2 = params.chunk(4, dim=-1)
            # Clamp scale to [0.5, 1.5] and shift to [-0.5, 0.5] for stability
            s1 = s1.clamp(-0.5, 0.5)
            b1 = b1.clamp(-0.5, 0.5)
            s2 = s2.clamp(-0.5, 0.5)
            b2 = b2.clamp(-0.5, 0.5)
            h = (1 + s1) * self.ln1(x) + b1
            x = x + self.retention(h)
            h = (1 + s2) * self.ln2(x) + b2
            x = x + self.ffn(h)
        else:
            # No feedback: standard pre-norm (pass 0, or non-CEDL eval)
            x = x + self.retention(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
        return x


class RetentionLayer(nn.Module):
    """Pre-norm retention layer with FFN."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.retention = MultiScaleRetention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.retention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================================
# CEDL Stages
# ============================================================================

class CStageRetention(nn.Module):
    """C-Stage: grid-cell-inspired periodic retention encoder."""

    def __init__(self, vocab: int, max_seq: int, d_model: int, n_heads: int,
                 n_layers: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        # Use periodic retention (grid cell kernels) instead of plain RetNet
        self.layers = nn.ModuleList([
            PeriodicRetentionLayer(d_model, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor,
                feedback: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        # Feedback modulates each layer via hierarchical neuromodulatory gating
        # (not additive at input — each layer's LN is modulated by feedback)
        for layer in self.layers:
            x = layer(x, feedback=feedback)
        return self.ln(x)


class EStage(nn.Module):
    """DG: expansion + pattern separation via AHSD + CSR.

    Anti-Hebbian Support Drift (AHSD): Tracks per-neuron activation
    frequencies and suppresses overused neurons before top-k selection.
    Prevents mode collapse and ensures full use of the sparse code space.
    Inspired by homeostatic plasticity and intrinsic excitability regulation
    that maintain the DG's characteristically low population activity
    (~2-5% of granule cells active; Chawla et al. 2005).

    Contrastive Support Repulsion (CSR): Auxiliary loss that penalizes
    support overlap between similar inputs. The E-stage doesn't just
    sparsify — it actively maximizes code dissimilarity for similar
    inputs, which is the computational definition of pattern separation.
    """

    def __init__(self, d_model: int, expansion: int = 4,
                 sparsity: float = 0.10, ema_decay: float = 0.99,
                 inhibition_strength: float = 0.5):
        super().__init__()
        self.d_expand = d_model * expansion
        self.k = max(1, int(self.d_expand * sparsity))
        self.expand = nn.Linear(d_model, self.d_expand)
        self.contract = nn.Linear(self.d_expand, d_model)
        self.ln = nn.LayerNorm(d_model)

        # AHSD: running frequency statistics for anti-Hebbian inhibition
        self.register_buffer('neuron_freq', torch.zeros(self.d_expand))
        self.ema_decay = ema_decay
        self.inhibition_strength = inhibition_strength
        # Learnable inhibition temperature
        self.inhibition_temp = nn.Parameter(torch.tensor(1.0))

    def set_sparsity(self, sparsity: float):
        """Update sparsity level (for annealing during training)."""
        self.k = max(1, int(self.d_expand * sparsity))

    def forward(self, h_C: torch.Tensor,
                feedback: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Inject D->E feedback if available (Loop 2 return path)
        if feedback is not None:
            h_C = h_C + feedback
        expanded = F.gelu(self.expand(h_C))

        # AHSD: suppress overused neurons before top-k
        if self.training and self.neuron_freq.sum() > 0:
            penalty = (self.neuron_freq * self.inhibition_strength
                       * F.softplus(self.inhibition_temp))
            inhibited = expanded - penalty
        else:
            inhibited = expanded

        # Top-k on inhibited activations, but use ORIGINAL values
        topk_vals, topk_idx = inhibited.topk(self.k, dim=-1)
        original_vals = expanded.gather(-1, topk_idx)
        sparse = torch.zeros_like(expanded)
        sparse.scatter_(-1, topk_idx, original_vals)

        # Update AHSD frequency statistics
        if self.training:
            with torch.no_grad():
                batch_freq = (sparse != 0).float().mean(dim=(0, 1))
                self.neuron_freq.mul_(self.ema_decay).add_(
                    batch_freq, alpha=1 - self.ema_decay)

        contracted = self.contract(sparse)
        return self.ln(contracted + h_C), sparse

    @staticmethod
    def csr_loss(h_C: torch.Tensor, sparse: torch.Tensor,
                 n_sample: int = 128) -> torch.Tensor:
        """Contrastive Support Repulsion: penalize support overlap between
        similar inputs. This IS pattern separation as an explicit objective.

        Args:
            h_C: [B, T, d] pre-expansion input
            sparse: [B, T, d_expand] sparse codes
            n_sample: subsample tokens for O(n^2) efficiency
        """
        B, T, _ = h_C.shape
        # Subsample for efficiency
        if T > n_sample:
            idx = torch.randperm(T, device=h_C.device)[:n_sample]
            h_sub = h_C[:, idx]
            s_sub = sparse[:, idx]
        else:
            h_sub, s_sub = h_C, sparse
            n_sample = T

        # Input similarity
        h_norm = F.normalize(h_sub, dim=-1)
        input_sim = F.relu(torch.bmm(h_norm, h_norm.transpose(1, 2)))

        # Support overlap
        mask = (s_sub != 0).float()
        k = mask.sum(dim=-1, keepdim=True).clamp(min=1)
        mask_norm = mask / k.sqrt()
        overlap = torch.bmm(mask_norm, mask_norm.transpose(1, 2))

        # Penalize: high input similarity × high support overlap
        eye = torch.eye(n_sample, device=h_C.device).unsqueeze(0)
        loss = ((input_sim * overlap) * (1 - eye)).sum()
        return loss / (B * n_sample * max(n_sample - 1, 1))


class DStage(nn.Module):
    """CA3: dual-memory — attractor refinement + explicit memory bank + Loop 2.

    Combines two complementary memory mechanisms, matching real CA3:
    1. Attractor refinement (implicit): patterns stored in shared QKV weights,
       retrieved via iterative self-attention settling to fixed points.
       Analog: CA3 recurrent collateral autoassociation.
    2. Memory bank (explicit): learned key-value slots accessed via
       cross-attention after attractor settling.
       Analog: CA3 specific learned synaptic associations.

    The attractor completes patterns from partial cues (implicit recall).
    The memory bank stores specific associations (explicit lookup).
    A learned gate integrates both sources.
    """

    def __init__(self, d_model: int, d_expand: int, n_heads: int,
                 refine_steps: int = 3, num_slots: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.refine_steps = refine_steps
        self.scale = self.d_head ** -0.5

        # === Implicit memory: attractor self-attention (CA3 recurrent collaterals) ===
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_out = nn.Linear(d_model, d_model)

        # Learned step sizes per refinement step (adaptive settling)
        self.step_sizes = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(refine_steps)])
        self.step_lns = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(refine_steps)])

        self.dropout = nn.Dropout(dropout)

        # === Explicit memory: learned key-value bank (CA3 synaptic associations) ===
        self.mem_keys = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.mem_vals = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln_mem = nn.LayerNorm(d_model)

        # Gate: blend attractor output with memory bank retrieval
        self.mem_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )

        # Loop 2: D->E feedback attention
        self.loop2_q_proj = nn.Linear(d_model, d_model)
        self.loop2_k_proj = nn.Linear(d_expand, d_model)
        self.loop2_v_proj = nn.Linear(d_expand, d_model)
        self.loop2_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.loop2_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.ln_loop2 = nn.LayerNorm(d_model)

    def attractor_settle(self, h: torch.Tensor,
                         causal: torch.Tensor) -> torch.Tensor:
        """Iterative attractor settling via self-attention with learned step sizes."""
        B, T, D = h.shape
        q_state = h
        for step in range(self.refine_steps):
            Q = self.w_q(q_state).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            K = self.w_k(q_state).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
            V = self.w_v(q_state).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

            attn = (Q @ K.transpose(-1, -2)) * self.scale
            attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), -1e9)
            attn = self.dropout(F.softmax(attn, dim=-1))
            attn_out = (attn @ V).transpose(1, 2).contiguous().view(B, T, D)
            attn_out = self.w_out(attn_out)

            eta = torch.sigmoid(self.step_sizes[step])
            q_state = self.step_lns[step](q_state + eta * attn_out)
        return q_state

    def forward(self, h_E: torch.Tensor,
                h_E_sparse: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = h_E.size(0), h_E.size(1)
        causal = torch.triu(
            torch.ones(T, T, device=h_E.device, dtype=torch.bool), 1)

        # Step 1: Implicit memory — attractor settling (CA3 recurrent collaterals)
        q_attractor = self.attractor_settle(h_E, causal)

        # Step 2: Explicit memory — cross-attention to memory bank (CA3 synaptic assoc.)
        mk = self.mem_keys.unsqueeze(0).expand(B, -1, -1)
        mv = self.mem_vals.unsqueeze(0).expand(B, -1, -1)
        retrieved, _ = self.cross_attn(q_attractor, mk, mv)
        q_mem = self.ln_mem(q_attractor + retrieved)

        # Step 3: Gate — blend implicit (attractor) and explicit (memory bank)
        gate = self.mem_gate(torch.cat([q_attractor, q_mem], dim=-1))
        q = gate * q_mem + (1 - gate) * q_attractor

        # Loop 2: D attends back to E ("does this episode match memory?")
        loop2_q = self.loop2_q_proj(q)
        loop2_k = self.loop2_k_proj(h_E_sparse)
        loop2_v = self.loop2_v_proj(h_E_sparse)
        loop2_out, _ = self.loop2_attn(
            loop2_q, loop2_k, loop2_v, attn_mask=causal)
        gate = self.loop2_gate(torch.cat([q, loop2_out], dim=-1))
        h_D = self.ln_loop2(q + gate * loop2_out)

        return h_D, loop2_out


class LStage(nn.Module):
    """CA1: two-channel comparator."""

    def __init__(self, d_model: int, vocab: int, n_heads: int,
                 dropout: float = 0.1):
        super().__init__()
        self.mem_head = nn.Linear(d_model, vocab)
        self.per_head = nn.Linear(d_model, vocab, bias=False)  # weight-tied

        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Learned fusion gate for CA1 two-stream integration
        self.fuse_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )

        self.loop1_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.loop1_gate = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.Sigmoid(),
        )
        self.ln_loop1 = nn.LayerNorm(d_model)

    def forward(self, h_D: torch.Tensor, h_C: torch.Tensor):
        # Only compute logits_mem here; logits_per is deferred to after
        # neuromodulation to avoid keeping multiple vocab-sized tensors alive.
        logits_mem = self.mem_head(h_D)

        # Loop 1 feedback: L attends to C
        # Learned fusion of memory (h_D) and perception (h_C) streams
        g = self.fuse_gate(torch.cat([h_D, h_C], dim=-1))
        fused = g * h_D + (1 - g) * h_C
        T = h_C.size(1)
        causal = torch.triu(torch.ones(T, T, device=h_C.device, dtype=torch.bool), 1)
        attn_out, _ = self.loop1_attn(fused, h_C, h_C, attn_mask=causal)
        gate = self.loop1_gate(torch.cat([fused, attn_out], dim=-1))
        loop1_fb = self.ln_loop1(fused + gate * attn_out)

        return logits_mem, loop1_fb


class NeuromodulatorGate(nn.Module):
    """Output-level gain modulation of C-stage by L and D signals.

    Distinct from the per-layer cholinergic AdaLN in PeriodicRetentionLayer:
    this gate models dopaminergic/noradrenergic modulation that gates the
    final hippocampal output (Lisman & Grace 2005), while AdaLN models
    cholinergic modulation during encoding (Hasselmo 1999).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.gain_from_L = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.gain_from_D = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.bias_from_L = nn.Linear(d_model, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, h_C, loop1_fb, h_D):
        g_L = self.gain_from_L(loop1_fb)
        g_D = self.gain_from_D(h_D)
        bias = self.bias_from_L(loop1_fb)
        return self.ln(g_L * g_D * h_C + bias)


def sigreg_loss(z: torch.Tensor) -> torch.Tensor:
    """Soft variance regularizer for D-stage outputs. Adapted from
    LeWorldModel's SIGReg (arXiv:2603.19312), but softened: only penalizes
    variance COLLAPSE (< 0.1), not high variance. This prevents
    representational collapse without fighting attractor basin structure
    (which is inherently multimodal/high-variance).
    """
    if z.dim() == 3:
        z = z.reshape(-1, z.size(-1))
    var = z.var(dim=0)
    # Only penalize collapse (var < 0.1), not high variance
    return F.relu(0.1 - var).mean()


def predictive_latent_loss(h_D: torch.Tensor, h_C: torch.Tensor) -> torch.Tensor:
    """JEPA-inspired predictive loss: D-stage output at position t should
    predict C-stage encoding at position t+1 in representation space.

    This makes the D-stage a forward model (not just a pattern completer),
    directly implementing the hippocampal predictive coding hypothesis:
    CA3 generates a prediction via Schaffer collaterals, CA1 compares it
    against actual EC input (Hasselmo 2005, Lisman & Redish 2009).

    Stop-gradient on target prevents representational collapse (JEPA principle).
    """
    pred = h_D[:, :-1, :]             # [B, T-1, d]
    target = h_C[:, 1:, :].detach()   # [B, T-1, d] — STOP GRADIENT
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    return F.smooth_l1_loss(pred_n, tgt_n)


class CEDLTwoLoop100M(nn.Module):
    """CEDL Two-Loop at 100M scale with coupled feedback loops.

    Forward pass:
        Pass 0 (feedforward): C -> E -> D -> L (no feedback)
        Pass 1..N (feedback): L->C and D->E signals enrich re-encoding
        Final: NeuromodulatorGate applies supplementary gain modulation

    Auxiliary losses (training only):
        - CSR: contrastive support repulsion for E-stage pattern separation
        - SIGReg: Gaussian regularizer on D-stage outputs (replaces attractor contrastive)
        - Predictive latent: D(t) predicts C(t+1) — JEPA-inspired forward model
    """

    def __init__(self, cfg: CEDLConfig, vocab: int, max_seq: int):
        super().__init__()
        d = cfg.d_model
        d_expand = d * cfg.e_expand
        self.n_feedback_iters = cfg.n_feedback_iters
        self.feedback_decay = cfg.feedback_decay
        # Feedback alpha: ramped 0→1 during warmup, then 1.0
        self.register_buffer('feedback_alpha', torch.tensor(0.0))

        self.c_stage = CStageRetention(
            vocab, max_seq, d, cfg.n_heads, cfg.c_layers, cfg.ffn_dim, cfg.dropout)
        self.e_stage = EStage(d, cfg.e_expand, cfg.e_sparsity)
        self.d_stage = DStage(d, d_expand, cfg.n_heads,
                              cfg.d_refine, cfg.d_slots, cfg.dropout)
        self.l_stage = LStage(d, vocab, cfg.n_heads, cfg.dropout)
        self.modulator = NeuromodulatorGate(d)

        # Weight tying: BOTH heads share C-stage embeddings
        # mem_head keeps its own bias for channel distinction
        self.l_stage.per_head.weight = self.c_stage.tok_emb.weight
        self.l_stage.mem_head.weight = self.c_stage.tok_emb.weight

    def forward(self, ids: torch.Tensor):
        # === Pass 0: feedforward (no feedback) ===
        h_C = self.c_stage(ids, feedback=None)
        h_E, h_E_sparse = self.e_stage(h_C, feedback=None)
        h_D, loop2_signal = self.d_stage(h_E, h_E_sparse)
        logits_mem, loop1_fb = self.l_stage(h_D, h_C)

        # Save Pass 0 outputs for aux losses (stable, no feedback dependency)
        h_C_p0, h_E_sparse_p0, h_D_p0 = h_C, h_E_sparse, h_D

        # === Feedback iterations ===
        alpha = self.feedback_alpha.item()
        for i in range(self.n_feedback_iters):
            decay = self.feedback_decay ** i
            fb_scale = alpha * decay

            if fb_scale > 0:
                h_C = self.c_stage(ids, feedback=fb_scale * loop1_fb)
                h_E, h_E_sparse = self.e_stage(h_C, feedback=fb_scale * loop2_signal)
                h_D, loop2_signal = self.d_stage(h_E, h_E_sparse)
                logits_mem, loop1_fb = self.l_stage(h_D, h_C)

        # Auxiliary losses on PASS 0 outputs (not feedback-modified)
        # This prevents the moving-target instability when feedback activates
        aux_loss = torch.tensor(0.0, device=ids.device)
        if self.training:
            aux_loss = aux_loss + 0.05 * self.e_stage.csr_loss(h_C_p0, h_E_sparse_p0)
            aux_loss = aux_loss + 0.1 * sigreg_loss(h_D_p0)
            aux_loss = aux_loss + 0.05 * predictive_latent_loss(h_D_p0, h_C_p0)
        del h_C_p0, h_E_sparse_p0, h_D_p0

        # Free intermediates
        del h_E, h_E_sparse

        # Neuromodulatory gating: supplementary gain modulation
        h_C_mod = self.modulator(h_C, loop1_fb, h_D)
        del loop1_fb
        logits_per_mod = self.l_stage.per_head(h_C_mod)
        gate_alpha = self.l_stage.gate_net(torch.cat([h_D, h_C_mod], dim=-1))
        del h_C, h_C_mod

        # In-place fusion (memory-efficient)
        logits_per_mod.mul_(1 - gate_alpha)
        logits_mem.mul_(gate_alpha)
        logits_per_mod.add_(logits_mem)
        del logits_mem, h_D

        return logits_per_mod, aux_loss


# ============================================================================
# Transformer (GPT-2 style)
# ============================================================================

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(x, x, x, attn_mask=mask)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ffn(self.ln2(x))
        return x


class TransformerLM100M(nn.Module):
    """GPT-2-style causal Transformer at 100M scale."""

    def __init__(self, cfg: TransformerConfig, vocab: int, max_seq: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(max_seq, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        # Weight tying
        self.head.weight = self.tok_emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        mask = torch.triu(torch.ones(T, T, device=ids.device, dtype=torch.bool), 1)
        for block in self.blocks:
            x = block(x, mask)
        return self.head(self.ln_f(x))


# ============================================================================
# Transformer-XL (segment-level recurrence)
# ============================================================================

class TransformerXLLM100M(nn.Module):
    """Transformer-XL at 100M scale with segment-level recurrence.

    Note: Uses learned absolute position embeddings (applied only to the
    current segment) rather than the relative position encodings of the
    original Dai et al. 2019 paper. This is a simplification — memory
    tokens carry positional information from their original segment.

    The attention mask gives memory tokens full visibility while maintaining
    causal masking within the current segment.
    """

    def __init__(self, cfg: TransformerXLConfig, vocab: int, max_seq: int):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(max_seq, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.tok_emb.weight
        self.mems: List[Optional[torch.Tensor]] = [None] * cfg.n_layers

    def reset_memory(self):
        self.mems = [None] * self.cfg.n_layers

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))

        for i, block in enumerate(self.blocks):
            mem = self.mems[i]
            if mem is not None:
                M = mem.size(1)
                cat = torch.cat([mem, x], dim=1)   # [B, M+T, d]
            else:
                M = 0
                cat = x
            S = M + T
            # Mask: memory fully visible, current segment causal
            # True = ignore for nn.MHA
            mask = torch.zeros(S, S, device=ids.device, dtype=torch.bool)
            if T > 1:
                mask[M:, M:] = torch.triu(
                    torch.ones(T, T, device=ids.device, dtype=torch.bool), 1)

            if self.training:
                out = torch.utils.checkpoint.checkpoint(
                    block, cat, mask, use_reentrant=False)
            else:
                out = block(cat, mask)
            x = out[:, -T:]  # keep only current segment output

            # Store current segment hidden states as memory (truncated to mem_len)
            with torch.no_grad():
                self.mems[i] = x[:, -self.cfg.mem_len:].detach()

        return self.head(self.ln_f(x))


# ============================================================================
# RetNet (standalone LM)
# ============================================================================

class RetNetLM100M(nn.Module):
    """Multi-scale retention LM at 100M scale.

    Uses learned absolute position embeddings instead of the original RetNet
    paper's xPos. The decay matrix provides temporal weighting; absolute
    embeddings provide positional discrimination — a simpler but effective
    substitute for a benchmark comparison.
    """

    def __init__(self, cfg: RetNetConfig, vocab: int, max_seq: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, cfg.d_model)
        self.pos_emb = nn.Embedding(max_seq, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([
            RetentionLayer(cfg.d_model, cfg.n_heads, cfg.ffn_dim, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(T, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))


# ============================================================================
# Mamba (Selective SSM — pure PyTorch for TPU compatibility)
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used by Mamba)."""
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SelectiveSSM(nn.Module):
    """Selective state-space model (S6) — pure PyTorch implementation.

    Faithful to Gu & Dao 2023: per-channel delta via low-rank dt_proj,
    input-dependent B/C, S4D-Lin initialization for A, parallel scan.

    Core: h[t] = A_bar[t] * h[t-1] + B_bar[t] * x[t]
          y[t] = C[t] * h[t]
    """

    def __init__(self, d_inner: int, d_state: int = 16, d_conv: int = 4,
                 dt_rank: Optional[int] = None):
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state
        if dt_rank is None:
            dt_rank = math.ceil(d_inner / 16)
        self.dt_rank = dt_rank

        # Conv1d for local context (causal padding)
        self.conv1d = nn.Conv1d(d_inner, d_inner, d_conv, padding=d_conv - 1,
                                groups=d_inner)

        # Input-dependent SSM params: dt (low-rank), B, C
        self.x_proj = nn.Linear(d_inner, dt_rank + d_state * 2, bias=False)
        # Low-rank dt projection (dt_rank -> d_inner, per-channel delta)
        self.dt_proj = nn.Linear(dt_rank, d_inner)
        # Initialize dt_proj bias: inverse_softplus(uniform(0.001, 0.1))
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A: S4D-Lin initialization (log-space for stability)
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)
                      .unsqueeze(0).expand(d_inner, -1).clone())
        )
        self.D = nn.Parameter(torch.ones(d_inner))  # skip connection

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_inner] -> y: [B, T, d_inner]"""
        B, T, D = x.shape
        N = self.d_state

        # Conv1d (causal)
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # Project to SSM params
        proj = self.x_proj(x_conv)                         # [B, T, dt_rank+2N]
        dt_input, B_input, C_input = proj.split(
            [self.dt_rank, N, N], dim=-1)
        delta = F.softplus(self.dt_proj(dt_input))         # [B, T, D] per-channel

        # Discretize via zero-order hold
        A = -torch.exp(self.A_log)                         # [D, N]
        A_bar = torch.exp(
            delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)
        )                                                  # [B, T, D, N]
        B_bar = delta.unsqueeze(-1) * B_input.unsqueeze(2) # [B, T, D, N]

        # Chunked sequential scan: h[t] = a[t]*h[t-1] + b[t]
        a = A_bar                                          # [B, T, D, N]
        b = B_bar * x_conv.unsqueeze(-1)                   # [B, T, D, N]
        h_all = self._chunked_scan(a, b)                     # [B, T, D, N]

        # Output: y[t] = C[t] * h[t]
        y = (h_all * C_input[:, :, None, :]).sum(-1)       # [B, T, D]
        y = y + x * self.D                                 # skip on raw input
        return y

    @staticmethod
    def _chunked_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Chunked sequential scan for linear recurrence h[t] = a[t]*h[t-1] + b[t].

        Sequential within chunks of 64 steps, state propagated between chunks.
        O(T) total iterations with good cache locality.

        Args:
            a: [B, T, D, N] multiplicative coefficients
            b: [B, T, D, N] additive terms
        Returns:
            h: [B, T, D, N] all hidden states
        """
        T = a.shape[1]
        CHUNK = 64
        h_list = []
        h_prev = torch.zeros_like(a[:, 0])  # [B, D, N]
        for start in range(0, T, CHUNK):
            end = min(start + CHUNK, T)
            a_chunk = a[:, start:end]
            b_chunk = b[:, start:end]
            h_chunk = []
            h = h_prev
            for t in range(end - start):
                h = a_chunk[:, t] * h + b_chunk[:, t]
                h_chunk.append(h)
            h_prev = h
            h_list.append(torch.stack(h_chunk, dim=1))

        return torch.cat(h_list, dim=1)[:, :T]


class MambaBlock(nn.Module):
    """Single Mamba block: in_proj -> SSM -> out_proj with gating."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2):
        super().__init__()
        d_inner = d_model * expand
        self.ln = RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)  # x and gate
        self.ssm = SelectiveSSM(d_inner, d_state, d_conv)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln(x)
        xz = self.in_proj(x)               # [B, T, 2*d_inner]
        x_ssm, z = xz.chunk(2, dim=-1)     # each [B, T, d_inner]
        x_ssm = self.ssm(x_ssm)
        x_ssm = x_ssm * F.silu(z)          # gating
        return residual + self.out_proj(x_ssm)


class MambaLM100M(nn.Module):
    """Mamba language model at 100M scale."""

    def __init__(self, cfg: MambaConfig, vocab: int, max_seq: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, cfg.d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(cfg.d_model, cfg.d_state, cfg.d_conv, cfg.expand)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(ids)
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


# ============================================================================
# LSTM Baseline
# ============================================================================

class LSTMLM100M(nn.Module):
    """LSTM language model at 100M scale."""

    def __init__(self, cfg: LSTMConfig, vocab: int, max_seq: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, cfg.d_model)
        self.lstm = nn.LSTM(
            cfg.d_model, cfg.hidden_size, cfg.n_layers,
            batch_first=True, dropout=cfg.dropout if cfg.n_layers > 1 else 0.0)
        # Initialize forget gate bias to 1.0 (Jozefowicz et al. 2015)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                # LSTM bias layout: [input_gate, forget_gate, cell_gate, output_gate]
                # Each gate has hidden_size entries; forget gate is the 2nd quarter
                n = param.size(0) // 4
                param.data[n:2*n].fill_(1.0)
        self.ln_f = nn.LayerNorm(cfg.hidden_size)
        # Project hidden_size -> d_model for weight tying
        self.proj = nn.Linear(cfg.hidden_size, cfg.d_model, bias=False)
        self.head = nn.Linear(cfg.d_model, vocab, bias=False)
        # Weight tying: head shares embedding weights
        self.head.weight = self.tok_emb.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(ids)
        x, _ = self.lstm(x)
        x = self.proj(self.ln_f(x))
        return self.head(x)


# ============================================================================
# Model Builder
# ============================================================================

ALL_MODELS = ["CEDL", "Transformer", "Transformer-XL", "RetNet", "Mamba"]


def build_model(tag: str, vocab: int, max_seq: int) -> nn.Module:
    """Build a ~100M parameter model by tag."""
    if tag == "CEDL":
        return CEDLTwoLoop100M(CEDLConfig(), vocab, max_seq)
    elif tag == "Transformer":
        return TransformerLM100M(TransformerConfig(), vocab, max_seq)
    elif tag == "Transformer-XL":
        return TransformerXLLM100M(TransformerXLConfig(), vocab, max_seq)
    elif tag == "RetNet":
        return RetNetLM100M(RetNetConfig(), vocab, max_seq)
    elif tag == "Mamba":
        return MambaLM100M(MambaConfig(), vocab, max_seq)
    elif tag == "LSTM":
        return LSTMLM100M(LSTMConfig(), vocab, max_seq)
    else:
        raise ValueError(f"Unknown model: {tag}")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def verify_all_params():
    """Print parameter counts for all models."""
    print("=" * 60)
    print("Parameter Verification (target: ~100M)")
    print("=" * 60)
    for tag in ALL_MODELS:
        model = build_model(tag, 50257, 1024)
        n = count_params(model)
        print(f"  {tag:20s}: {n:>12,d} ({n/1e6:.1f}M)")
        del model
    print("=" * 60)


# ============================================================================
# Data Pipeline
# ============================================================================

class TextChunkDataset(Dataset):
    """Pre-tokenized text split into fixed-length chunks."""

    def __init__(self, token_ids: torch.Tensor, chunk_size: int):
        self.chunk_size = chunk_size
        n_chunks = len(token_ids) // chunk_size
        self.data = token_ids[:n_chunks * chunk_size].view(n_chunks, chunk_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_data(cfg: Config):
    """Load and tokenize dataset. Returns train/val/test DataLoaders."""
    from transformers import GPT2TokenizerFast
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    def tokenize_rows(rows):
        """Tokenize row-by-row to avoid OOM from giant string join."""
        all_ids = []
        for text in rows:
            if text.strip():
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_ids.extend(ids)
        return torch.tensor(all_ids, dtype=torch.long)

    if cfg.dataset == "wikitext103":
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1")
        print("Tokenizing train split...")
        train_ids = tokenize_rows(ds["train"]["text"])
        print("Tokenizing val split...")
        val_ids = tokenize_rows(ds["validation"]["text"])
        print("Tokenizing test split...")
        test_ids = tokenize_rows(ds["test"]["text"])
        del ds
    elif cfg.dataset == "c4":
        from datasets import load_dataset
        ds_train = load_dataset("allenai/c4", "en", split="train", streaming=True)
        ds_val = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        all_ids = []
        total_tokens = 0
        for example in ds_train:
            ids = tokenizer.encode(example["text"], add_special_tokens=False)
            all_ids.extend(ids)
            total_tokens += len(ids)
            if total_tokens > 400_000_000:
                break
        train_ids = torch.tensor(all_ids, dtype=torch.long)
        del all_ids
        val_ids_list = []
        val_tokens = 0
        for example in ds_val:
            ids = tokenizer.encode(example["text"], add_special_tokens=False)
            val_ids_list.extend(ids)
            val_tokens += len(ids)
            if val_tokens > 2_000_000:
                break
        val_ids = torch.tensor(val_ids_list, dtype=torch.long)
        del val_ids_list
        test_ids = val_ids  # C4 has no official test; reuse val
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    # Chunk size = max_seq + 1 (input + target)
    chunk = cfg.max_seq + 1
    train_ds = TextChunkDataset(train_ids, chunk)
    val_ds = TextChunkDataset(val_ids, chunk)
    test_ds = TextChunkDataset(test_ids, chunk)

    pin = not cfg.tpu  # pin_memory only helps CUDA, not XLA
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=4, pin_memory=pin, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin, drop_last=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin, drop_last=True,
                             persistent_workers=True)

    print(f"Dataset: {cfg.dataset}")
    print(f"  Train chunks: {len(train_ds):,}")
    print(f"  Val chunks:   {len(val_ds):,}")
    print(f"  Test chunks:  {len(test_ds):,}")

    return train_loader, val_loader, test_loader, tokenizer


def make_loaders(train_ds, val_ds, test_ds, batch_size: int, tpu: bool = False):
    """Build DataLoaders from pre-tokenized datasets (fast, no re-tokenization)."""
    pin = not tpu
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=pin, drop_last=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=2, pin_memory=pin, drop_last=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=pin, drop_last=True,
                             persistent_workers=True)
    return train_loader, val_loader, test_loader


# ============================================================================
# Learning Rate Schedule
# ============================================================================

def get_lr(step: int, cfg: Config) -> float:
    """Cosine schedule with linear warmup."""
    if step < cfg.warmup_steps:
        return cfg.lr * step / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


# ============================================================================
# Training Loop
# ============================================================================

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device,
             max_steps: int = 200, tpu: bool = False) -> float:
    """Compute cross-entropy loss → perplexity."""
    _xm = None
    if tpu:
        import torch_xla.core.xla_model as _xm
    model.eval()
    # Reset Transformer-XL memory for clean evaluation
    if hasattr(model, 'reset_memory'):
        model.reset_memory()
    total_loss = 0.0
    count = 0
    for i, batch in enumerate(loader):
        if i >= max_steps:
            break
        batch = batch.to(device)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(input_ids)
        if isinstance(logits, tuple):
            logits = logits[0]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               targets.reshape(-1))
        total_loss += loss.item()
        del logits, loss
        count += 1
        if _xm is not None:
            _xm.mark_step()
    model.train()
    avg_loss = total_loss / max(count, 1)
    return math.exp(avg_loss)


def train(model: nn.Module, tag: str, cfg: Config, device: torch.device,
          train_loader: DataLoader, val_loader: DataLoader):
    """Main training loop with gradient accumulation and cosine LR."""
    # Import XLA utilities if on TPU
    xm = None
    if cfg.tpu:
        import torch_xla.core.xla_model as xm

    model.train()

    # Differential learning rates for CEDL (CLS: C=slow, E/D/L=fast)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model
    is_cedl = isinstance(raw_model, CEDLTwoLoop100M)
    if is_cedl:
        c_param_ids = {id(p) for p in raw_model.c_stage.parameters()}
        c_params = list(raw_model.c_stage.parameters())
        edl_params = [p for p in raw_model.parameters()
                      if id(p) not in c_param_ids]
        optimizer = torch.optim.AdamW([
            {"params": c_params, "lr": cfg.lr},
            {"params": edl_params, "lr": cfg.lr * 1.5},
        ], weight_decay=cfg.weight_decay, betas=(0.9, 0.95))
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95))

    # GradScaler is unnecessary with bfloat16 (same exponent range as fp32).
    # Only use it for fp16 mixed precision on CUDA.
    scaler = None

    step = 0
    accum_loss = 0.0
    best_val_ppl = float('inf')
    os.makedirs(cfg.save_dir, exist_ok=True)

    print(f"\nTraining {tag} ({count_params(model)/1e6:.1f}M params)")
    print(f"  Device: {device}")
    print(f"  Effective batch: {cfg.batch_size * cfg.grad_accum} seqs "
          f"= {cfg.batch_size * cfg.grad_accum * cfg.max_seq / 1e3:.0f}K tokens")
    print(f"  Max steps: {cfg.max_steps:,}")

    t0 = time.time()
    data_iter = iter(train_loader)
    cedl_cfg = CEDLConfig() if is_cedl else None

    while step < cfg.max_steps:
        optimizer.zero_grad()

        for micro in range(cfg.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                # Reset TXL memory at epoch boundary to avoid stale
                # cross-epoch memory leakage (data is shuffled)
                if hasattr(model, 'reset_memory'):
                    model.reset_memory()
                data_iter = iter(train_loader)
                batch = next(data_iter)

            batch = batch.to(device)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            # On TPU, autocast with device_type='xla'; on GPU, 'cuda'
            autocast_device = 'xla' if cfg.tpu else device.type
            with torch.amp.autocast(autocast_device, dtype=torch.bfloat16,
                                    enabled=cfg.bfloat16):
                output = model(input_ids)
                # CEDL returns (logits, aux_loss); others return logits
                if isinstance(output, tuple):
                    logits, aux_loss = output[0], output[1]
                else:
                    logits, aux_loss = output, torch.tensor(0.0, device=device)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                       targets.reshape(-1))
                loss = (loss + aux_loss) / cfg.grad_accum

            loss.backward()
            accum_loss += loss.item()
            del logits, loss
            # Mark step after each micro-batch to prevent graph accumulation
            if xm is not None:
                xm.mark_step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        # LR schedule (with differential rates for CEDL)
        lr = get_lr(step, cfg)
        for i, pg in enumerate(optimizer.param_groups):
            multiplier = 1.0 if (not is_cedl or i == 0) else 1.5
            pg['lr'] = lr * multiplier

        if xm is not None:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()

        step += 1

        # CEDL feedback warmup + sparsity annealing
        if is_cedl:
            # Feedback warmup: alpha ramps 0→1 over warmup window
            if step < cedl_cfg.feedback_warmup_start:
                raw_model.feedback_alpha.fill_(0.0)
            elif step < cedl_cfg.feedback_warmup_end:
                frac = (step - cedl_cfg.feedback_warmup_start) / (
                    cedl_cfg.feedback_warmup_end - cedl_cfg.feedback_warmup_start)
                raw_model.feedback_alpha.fill_(frac)
            else:
                raw_model.feedback_alpha.fill_(1.0)
            # Sparsity annealing: 10% -> 5% after 80% of training
            anneal_step = int(cfg.max_steps * cedl_cfg.sparsity_anneal_frac)
            if step >= anneal_step and cfg.max_steps > anneal_step:
                frac = (step - anneal_step) / (cfg.max_steps - anneal_step)
                sparsity = cedl_cfg.e_sparsity + frac * (
                    cedl_cfg.sparsity_final - cedl_cfg.e_sparsity)
                raw_model.e_stage.set_sparsity(sparsity)

        # Logging
        if step % 100 == 0:
            elapsed = time.time() - t0
            tokens_per_sec = (step * cfg.batch_size * cfg.grad_accum * cfg.max_seq) / elapsed
            print(f"  [{step:>6d}/{cfg.max_steps}] loss={accum_loss/100:.4f} "
                  f"lr={lr:.2e} tok/s={tokens_per_sec:.0f}")
            accum_loss = 0.0

        # Eval
        if step % cfg.eval_interval == 0:
            val_ppl = evaluate(model, val_loader, device, cfg.eval_steps, tpu=cfg.tpu)
            print(f"  >>> Val PPL: {val_ppl:.2f}")
            if val_ppl < best_val_ppl:
                best_val_ppl = val_ppl
                ckpt_path = os.path.join(cfg.save_dir, f"{tag}_best.pt")
                if xm is not None:
                    xm.save(model.state_dict(), ckpt_path)
                else:
                    torch.save(model.state_dict(), ckpt_path)
                print(f"  >>> Saved best checkpoint: {ckpt_path}")

        # Periodic save
        if step % cfg.save_interval == 0:
            ckpt_path = os.path.join(cfg.save_dir, f"{tag}_step{step}.pt")
            if xm is not None:
                xm.save(model.state_dict(), ckpt_path)
            else:
                torch.save(model.state_dict(), ckpt_path)

    return best_val_ppl


# ============================================================================
# Structured Benchmark (A1-A5 Comprehension + B1-B4 Math + C1-C3 CEDL-probe)
# ============================================================================

def run_structured_benchmark(model, tokenizer, device, max_seq=1024, tpu=False):
    """Zero-shot structured benchmark: 12 tasks x 4 difficulty levels."""
    import random as _rng
    _rng.seed(42)  # fixed seed for reproducibility

    xm_mod = None
    if tpu:
        try:
            import torch_xla.core.xla_model as _xm
            xm_mod = _xm
        except ImportError:
            pass

    N_INSTANCES = 200

    NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace",
             "Hank", "Iris", "Jack", "Kate", "Leo", "Mia", "Nick",
             "Olivia", "Paul", "Quinn", "Rosa", "Sam", "Tina"]
    COLORS = ["red", "blue", "green", "pink", "gray", "brown", "white",
              "black", "gold", "silver", "orange", "purple", "yellow"]
    OBJECTS = ["car", "hat", "bag", "pen", "cup", "box", "ring",
               "lamp", "book", "ball", "shoe", "coat", "desk", "bell",
               "fork", "drum", "fish", "kite", "coin", "vase"]
    TRAITS = ["tall", "short", "fast", "slow", "kind", "bold",
              "calm", "warm", "cold", "rich", "poor", "wise",
              "young", "old", "brave"]
    ANIMALS = ["dog", "cat", "bird", "fish", "frog", "bear", "deer",
               "wolf", "duck", "fox", "pig", "cow", "ant", "bee", "rat"]
    FRUITS = ["apple", "banana", "grape", "lemon", "mango", "peach",
              "plum", "cherry", "melon", "berry"]

    def gen_A1(level):
        n = [2, 4, 6, 8][level]
        names = _rng.sample(NAMES, n)
        colors = [_rng.choice(COLORS) for _ in range(n)]
        objs = [_rng.choice(OBJECTS) for _ in range(n)]
        facts = ". ".join(f"{names[i]} has a {colors[i]} {objs[i]}" for i in range(n))
        qi = _rng.randint(0, n - 1)
        return f"{facts}. Who has a {colors[qi]} {objs[qi]}? Answer:", f" {names[qi]}"

    def gen_A2(level):
        n = [2, 4, 6, 8][level]
        names = _rng.sample(NAMES, n)
        traits = [_rng.choice(TRAITS) for _ in range(n)]
        facts = ". ".join(f"{names[i]} is {traits[i]}" for i in range(n))
        qi = _rng.randint(0, n - 1)
        is_true = _rng.random() < 0.5
        ct = traits[qi] if is_true else _rng.choice([t for t in TRAITS if t != traits[qi]])
        return f"{facts}. Claim: {names[qi]} is {ct}. True or false? Answer:", " true" if is_true else " false"

    def gen_A3(level):
        n_hops = [1, 2, 3, 3][level]
        n_dist = [0, 0, 0, 3][level]
        all_n = _rng.sample(NAMES, n_hops + 1 + n_dist * 2)
        chain = all_n[:n_hops + 1]
        rels = [f"{chain[i]} is parent of {chain[i+1]}" for i in range(n_hops)]
        for i in range(n_dist):
            rels.append(f"{all_n[n_hops+1+2*i]} is friend of {all_n[n_hops+2+2*i]}")
        _rng.shuffle(rels)
        label = {1: "child", 2: "grandchild", 3: "great grandchild"}
        return f"{'. '.join(rels)}. Who is {chain[0]}'s {label[n_hops]}? Answer:", f" {chain[-1]}"

    def gen_A4(level):
        n_pairs, noise = 4, [0, 2, 4, 8][level]
        keys = _rng.sample(FRUITS, n_pairs)
        vals = _rng.sample(COLORS, n_pairs)
        parts = []
        for i in range(n_pairs):
            parts.append(f"key: {keys[i]} value: {vals[i]}.")
            if noise > 0:
                parts.append(" ".join(_rng.choice(ANIMALS) for _ in range(noise)) + ".")
        qi = _rng.randint(0, n_pairs - 1)
        return f"{' '.join(parts)} What is the value of {keys[qi]}? Answer:", f" {vals[qi]}"

    def gen_B1(level):
        if level == 0:
            a, b = _rng.randint(1, 9), _rng.randint(1, 9); r = a + b
            return f"What is {a} + {b}? Answer:", f" {r}"
        elif level == 1:
            a, b = _rng.randint(10, 49), _rng.randint(10, 49); r = a + b
            return f"What is {a} + {b}? Answer:", f" {r}"
        elif level == 2:
            a, b = _rng.randint(1, 30), _rng.randint(1, 30)
            op = _rng.choice(["+", "-"])
            if op == "-": a, b = max(a, b), min(a, b)
            r = (a + b) if op == "+" else (a - b)
            return f"What is {a} {op} {b}? Answer:", f" {r}"
        else:
            a, b, c = _rng.randint(1, 20), _rng.randint(1, 20), _rng.randint(1, 15)
            s1 = a + b; op2 = _rng.choice(["+", "-"])
            if op2 == "-" and s1 < c: c = _rng.randint(1, s1)
            r = (s1 + c) if op2 == "+" else (s1 - c)
            return f"What is {a} + {b} {op2} {c}? Answer:", f" {r}"

    def gen_B2(level):
        if level == 0:
            a, b = _rng.randint(1, 30), _rng.randint(1, 30)
            return f"If x = {a} + {b}, what is x? Answer:", f" {a+b}"
        elif level == 1:
            a, b, c = _rng.randint(1, 15), _rng.randint(1, 15), _rng.randint(1, 15)
            return f"If x = {a} + {b}, and y = x + {c}, what is y? Answer:", f" {a+b+c}"
        elif level == 2:
            a, b, c = _rng.randint(2, 9), _rng.randint(2, 9), _rng.randint(1, 15)
            return f"If x = {a} * {b} + {c}, what is x? Answer:", f" {a*b+c}"
        else:
            a, b, c, d = _rng.randint(1, 12), _rng.randint(1, 12), _rng.randint(1, 12), _rng.randint(1, 12)
            return f"If x = {a} + {b}, and y = {c} + {d}, what is x + y? Answer:", f" {a+b+c+d}"

    def gen_B3(level):
        if level == 0:
            a, b = _rng.randint(1, 99), _rng.randint(1, 99)
            while a == b: b = _rng.randint(1, 99)
            return f"Which is larger, {a} or {b}? Answer:", f" {max(a, b)}"
        elif level == 1:
            vals = _rng.sample(range(1, 99), 3)
            return f"Which is largest, {vals[0]}, {vals[1]}, or {vals[2]}? Answer:", f" {max(vals)}"
        elif level == 2:
            a, b, c, d = _rng.randint(1, 40), _rng.randint(1, 40), _rng.randint(1, 40), _rng.randint(1, 40)
            while a + b == c + d: d = _rng.randint(1, 40)
            return f"Which is larger, {a} + {b} or {c} + {d}? Answer:", f" {max(a+b, c+d)}"
        else:
            b = _rng.choice([3, 4, 5, 6, 7, 8, 9]); a = _rng.randint(10, 99)
            return f"What is {a} mod {b}? Answer:", f" {a % b}"

    def gen_B4(level):
        if level <= 2:
            target = _rng.choice(ANIMALS[:5])
            others = [x for x in ANIMALS[:8] if x != target]
            seq_len = [_rng.randint(4, 8), _rng.randint(6, 10), _rng.randint(8, 15)][level]
            seq = [_rng.choice([target] + others) for _ in range(seq_len)]
            r = sum(1 for s in seq if s == target)
            return f"Count the number of times '{target}' appears: {' '.join(seq)}. Answer:", f" {r}"
        else:
            t1, t2 = _rng.sample(ANIMALS[:6], 2)
            others = [x for x in ANIMALS[:10] if x not in (t1, t2)]
            seq = [_rng.choice([t1, t2] + others) for _ in range(_rng.randint(6, 12))]
            r = sum(1 for s in seq if s in (t1, t2))
            return f"Count the total times '{t1}' or '{t2}' appear: {' '.join(seq)}. Answer:", f" {r}"

    def gen_A5(level):
        """Memory update: present facts, then update some, query latest value."""
        n_facts, n_updates = [(2, 1), (4, 2), (6, 3), (8, 4)][level]
        names = _rng.sample(NAMES, n_facts)
        attrs = _rng.sample(OBJECTS, n_facts)
        # Original facts
        facts = ". ".join(f"{names[i]} owns a {attrs[i]}" for i in range(n_facts))
        # Choose entities to update
        update_idx = set(_rng.sample(range(n_facts), n_updates))
        new_attrs = {}
        for ui in update_idx:
            remaining = [o for o in OBJECTS if o != attrs[ui]]
            new_attrs[ui] = _rng.choice(remaining)
        # Update sentences
        updates = ". ".join(f"{names[ui]} now owns a {new_attrs[ui]}"
                            for ui in sorted(update_idx))
        # Query: 50/50 updated vs unchanged
        query_updated = _rng.random() < 0.5
        updated_list = list(update_idx)
        unchanged_list = [i for i in range(n_facts) if i not in update_idx]
        if query_updated and updated_list:
            qi = _rng.choice(updated_list)
        elif unchanged_list:
            qi = _rng.choice(unchanged_list)
        else:
            qi = _rng.choice(updated_list)
        answer = new_attrs[qi] if qi in update_idx else attrs[qi]
        return f"{facts}. Update: {updates}. What does {names[qi]} own now? Answer:", f" {answer}"

    # === C-tasks: probe novel CEDL mechanisms ===

    def gen_C1(level):
        """Periodic pattern detection: predict next element in a repeating sequence.
        Probes C-stage grid-cell periodic retention."""
        period, noise_per, length = [
            (2, 0, 8), (3, 0, 15), (4, 2, 20), (6, 3, 30)][level]
        # Build a repeating pattern from distinct tokens
        pattern_items = _rng.sample(OBJECTS[:10], period)
        sequence = []
        for i in range(length):
            sequence.append(pattern_items[i % period])
            if noise_per > 0 and _rng.random() < 0.3:
                for _ in range(min(noise_per, 2)):
                    sequence.append(_rng.choice(ANIMALS[:6]))
        # Query: what comes next?
        next_item = pattern_items[len([s for s in sequence if s in pattern_items]) % period]
        prompt = " ".join(sequence)
        return f"Repeating pattern: {prompt}. Next item:", f" {next_item}"

    def gen_C2(level):
        """Near-miss disambiguation: distinguish entities with overlapping attributes.
        Probes E-stage AHSD/CSR pattern separation."""
        n_entities, n_shared = [(2, 1), (2, 2), (3, 2), (4, 3)][level]
        names = _rng.sample(NAMES, n_entities)
        # Shared attributes (all entities have these)
        shared_colors = _rng.sample(COLORS, n_shared)
        shared_objs = _rng.sample(OBJECTS[:10], n_shared)
        # Distinguishing attribute (unique per entity)
        unique_objs = _rng.sample(OBJECTS[10:], n_entities)

        facts = []
        for i, name in enumerate(names):
            # Shared facts
            for j in range(n_shared):
                facts.append(f"{name} has a {shared_colors[j]} {shared_objs[j]}")
            # Unique fact
            facts.append(f"{name} has a {unique_objs[i]}")
        _rng.shuffle(facts)

        # Query the unique attribute
        qi = _rng.randint(0, n_entities - 1)
        return (f"{'. '.join(facts)}. Who has a {unique_objs[qi]}? Answer:",
                f" {names[qi]}")

    def gen_C3(level):
        """Partial cue completion: complete a degraded fact from stored patterns.
        Probes D-stage attractor pattern completion."""
        n_facts, n_missing = [(2, 1), (3, 1), (4, 2), (5, 2)][level]
        names = _rng.sample(NAMES, n_facts)
        colors = [_rng.choice(COLORS) for _ in range(n_facts)]
        objs = _rng.sample(OBJECTS, n_facts)
        traits = _rng.sample(TRAITS, n_facts)

        # Full facts
        facts = [f"{names[i]} is {traits[i]} and has a {colors[i]} {objs[i]}"
                 for i in range(n_facts)]
        context = ". ".join(facts)

        # Choose which fact to query via partial cue
        qi = _rng.randint(0, n_facts - 1)
        # Build partial cue with some attributes masked
        cue_parts = []
        query_attr = None
        attrs = [("trait", traits[qi]), ("color", colors[qi]), ("object", objs[qi])]
        # Randomly mask n_missing attributes, query one of them
        mask_indices = _rng.sample(range(len(attrs)), min(n_missing, len(attrs)))
        query_idx = _rng.choice(mask_indices)
        query_attr_name, query_attr_val = attrs[query_idx]

        # Build cue string
        cue = f"{names[qi]}"
        for j, (aname, aval) in enumerate(attrs):
            if j in mask_indices:
                cue += f" ??? {aname}"
            else:
                cue += f" {aval}"

        if query_attr_name == "color":
            return (f"{context}. Partial cue: {cue}. What is {names[qi]}'s color? Answer:",
                    f" {query_attr_val}")
        elif query_attr_name == "object":
            return (f"{context}. Partial cue: {cue}. What does {names[qi]} have? Answer:",
                    f" {query_attr_val}")
        else:  # trait
            return (f"{context}. Partial cue: {cue}. What trait does {names[qi]} have? Answer:",
                    f" {query_attr_val}")

    TASKS = {
        "A1_fact_retrieval": gen_A1, "A2_consistency": gen_A2,
        "A3_multihop": gen_A3, "A4_noisy_retrieval": gen_A4,
        "A5_memory_update": gen_A5,
        "B1_arithmetic": gen_B1, "B2_algebra": gen_B2,
        "B3_comparison": gen_B3, "B4_counting": gen_B4,
        "C1_periodic_pattern": gen_C1, "C2_near_miss": gen_C2,
        "C3_partial_completion": gen_C3,
    }

    # Build candidate answer sets for log-likelihood ranking.
    CANDIDATES = {
        "A1_fact_retrieval": NAMES,
        "A2_consistency": ["true", "false"],
        "A3_multihop": NAMES,
        "A4_noisy_retrieval": COLORS,
        "A5_memory_update": OBJECTS,
        "B1_arithmetic": [str(i) for i in range(-50, 200)],
        "B2_algebra": [str(i) for i in range(-50, 200)],
        "B3_comparison": [str(i) for i in range(0, 200)],
        "B4_counting": [str(i) for i in range(0, 30)],
        "C1_periodic_pattern": OBJECTS[:10],
        "C2_near_miss": NAMES,
        "C3_partial_completion": COLORS + OBJECTS + TRAITS,
    }

    # Pre-tokenize candidates (answer always starts with space)
    cand_ids_cache = {}
    for task_name, cands in CANDIDATES.items():
        cand_ids_cache[task_name] = []
        for c in cands:
            ids = tokenizer.encode(f" {c}", add_special_tokens=False)
            cand_ids_cache[task_name].append((c, ids))

    model.eval()
    if hasattr(model, 'reset_memory'):
        model.reset_memory()
    results = {}
    for task_name, gen_fn in TASKS.items():
        results[task_name] = {}
        cand_list = cand_ids_cache[task_name]
        for level in range(4):
            correct = 0
            for _ in range(N_INSTANCES):
                # Reset TXL memory between instances to prevent leakage
                if hasattr(model, 'reset_memory'):
                    model.reset_memory()
                prompt_str, answer_str = gen_fn(level)
                answer_str_stripped = answer_str.strip()
                prompt_ids = tokenizer.encode(prompt_str)
                if len(prompt_ids) >= max_seq:
                    prompt_ids = prompt_ids[-(max_seq - 1):]
                inp = torch.tensor([prompt_ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(inp)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                    # Log-probs at the last position
                    log_probs = F.log_softmax(logits[0, -1], dim=-1)
                    del logits
                    # Score each candidate by sum of log-probs of its tokens
                    best_score = float('-inf')
                    best_cand = None
                    for cand_str, cand_tok_ids in cand_list:
                        if not cand_tok_ids:
                            continue
                        # For multi-token answers, score first token only
                        # (full sequence scoring would need autoregressive passes)
                        score = log_probs[cand_tok_ids[0]].item()
                        if score > best_score:
                            best_score = score
                            best_cand = cand_str
                    if best_cand == answer_str_stripped:
                        correct += 1
                if xm_mod is not None:
                    xm_mod.mark_step()
            results[task_name][level] = correct / N_INSTANCES

    # Print table
    print(f"\n{'='*72}")
    print("Structured Benchmark Results (accuracy per task per level)")
    print(f"{'='*72}")
    header = f"{'Task':<22s}" + "".join(f" {'L'+str(i):>7s}" for i in range(4)) + f" {'Avg':>7s}"
    print(header)
    print("-" * len(header))
    a_accs, b_accs, c_accs = [], [], []
    for tn in TASKS:
        row = f"{tn:<22s}"
        avgs = []
        for lv in range(4):
            acc = results[tn][lv]
            row += f" {acc:>6.1%}"; avgs.append(acc)
        row += f" {sum(avgs)/4:>6.1%}"
        print(row)
        if tn.startswith("A"):
            a_accs.extend(avgs)
        elif tn.startswith("B"):
            b_accs.extend(avgs)
        else:
            c_accs.extend(avgs)
    print("-" * len(header))
    print(f"{'Content (A1-A5)':<22s}{' ':>31s} {sum(a_accs)/len(a_accs):>6.1%}")
    print(f"{'Math (B1-B4)':<22s}{' ':>31s} {sum(b_accs)/len(b_accs):>6.1%}")
    print(f"{'CEDL-probe (C1-C3)':<22s}{' ':>31s} {sum(c_accs)/len(c_accs):>6.1%}")
    all_accs = a_accs + b_accs + c_accs
    print(f"{'Overall':<22s}{' ':>31s} {sum(all_accs)/len(all_accs):>6.1%}")
    print(f"{'='*72}\n")
    return results


# ============================================================================
# Downstream Evaluation (lm-evaluation-harness wrapper)
# ============================================================================

def run_downstream_eval(model: nn.Module, tag: str, device: torch.device):
    """Zero-shot evaluation on LAMBADA, HellaSwag, ARC-Easy."""
    try:
        from lm_eval import evaluator
        from lm_eval.api.model import LM
        from transformers import GPT2TokenizerFast
    except ImportError:
        print("  lm-eval not installed. Skipping downstream eval.")
        print("  Install: pip install lm-eval transformers")
        return {}

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    class WrappedModel(LM):
        def __init__(self, model, tokenizer, device):
            super().__init__()
            self._model = model
            self._tokenizer = tokenizer
            self._device = device

        @property
        def eot_token_id(self):
            return self._tokenizer.eos_token_id

        @property
        def max_length(self):
            return 1024

        @property
        def max_gen_toks(self):
            return 256

        @property
        def batch_size(self):
            return 8

        @property
        def device(self):
            return self._device

        def tok_encode(self, string):
            return self._tokenizer.encode(string, add_special_tokens=False)

        def tok_decode(self, tokens):
            return self._tokenizer.decode(tokens)

        def loglikelihood(self, requests):
            results = []
            for req in requests:
                # Handle both tuple and Instance API (lm-eval v0.4+)
                if hasattr(req, 'args'):
                    ctx, cont = req.args
                else:
                    ctx, cont = req
                ctx_ids = self.tok_encode(ctx)
                cont_ids = self.tok_encode(cont)
                all_ids = ctx_ids + cont_ids
                # Truncate from LEFT to keep continuation intact
                if len(all_ids) > 1024:
                    excess = len(all_ids) - 1024
                    all_ids = all_ids[excess:]
                    ctx_ids = ctx_ids[excess:]
                input_ids = torch.tensor([all_ids], device=self._device)
                with torch.no_grad():
                    logits = self._model(input_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
                # Score continuation tokens
                start = len(ctx_ids)
                log_probs = F.log_softmax(logits[0], dim=-1)
                cont_log_prob = 0.0
                is_greedy = True
                for j in range(start, len(all_ids)):
                    tok = all_ids[j]
                    lp = log_probs[j - 1, tok].item()
                    cont_log_prob += lp
                    if logits[0, j - 1].argmax().item() != tok:
                        is_greedy = False
                results.append((cont_log_prob, is_greedy))
            return results

        def loglikelihood_rolling(self, requests):
            results = []
            for req in requests:
                if hasattr(req, 'args'):
                    (text,) = req.args
                else:
                    text = req[0] if isinstance(req, tuple) else req
                all_ids = self.tok_encode(text)
                # Sliding window with stride to preserve context.
                # Each window is 1024 tokens; stride=512 means 512 tokens
                # of overlap provide context for the scored tokens.
                max_len = 1024
                stride = 512
                total_lp = 0.0
                scored = set()  # track which positions we've scored
                for start in range(0, max(1, len(all_ids) - 1), stride):
                    end = min(start + max_len, len(all_ids))
                    chunk = all_ids[start:end]
                    if len(chunk) < 2:
                        continue
                    input_t = torch.tensor([chunk], device=self._device)
                    with torch.no_grad():
                        logits = self._model(input_t)
                        if isinstance(logits, tuple):
                            logits = logits[0]
                    log_probs = F.log_softmax(logits[0], dim=-1)
                    # Only score tokens in the non-overlap portion
                    # (except for the first window which scores from pos 1)
                    score_start = 1 if start == 0 else (end - start - stride)
                    for j in range(max(1, score_start), len(chunk)):
                        abs_pos = start + j
                        if abs_pos not in scored:
                            total_lp += log_probs[j - 1, chunk[j]].item()
                            scored.add(abs_pos)
                    if end == len(all_ids):
                        break
                results.append((total_lp,))
            return results

        def generate_until(self, requests):
            results = []
            for req in requests:
                # Handle both tuple and Instance API (lm-eval v0.4+)
                if hasattr(req, 'args'):
                    ctx, kwargs = req.args
                else:
                    ctx, kwargs = req
                input_ids = self.tok_encode(ctx)[-900:]
                input_t = torch.tensor([input_ids], device=self._device)
                max_gen = kwargs.get("max_gen_toks", 64)
                stop = kwargs.get("until", [])
                for _ in range(max_gen):
                    with torch.no_grad():
                        logits = self._model(input_t[:, -1024:])
                        if isinstance(logits, tuple):
                            logits = logits[0]
                    next_tok = logits[0, -1].argmax().item()
                    input_t = torch.cat([input_t,
                        torch.tensor([[next_tok]], device=self._device)], dim=1)
                    gen_text = self._tokenizer.decode(
                        input_t[0, len(input_ids):].tolist())
                    if any(s in gen_text for s in stop):
                        break
                results.append(gen_text)
            return results

    wrapped = WrappedModel(model, tokenizer, device)
    tasks = ["lambada_openai", "hellaswag", "arc_easy"]

    try:
        results = evaluator.simple_evaluate(
            model=wrapped, tasks=tasks, batch_size=8)
        print(f"\n  Downstream Results ({tag}):")
        for task_name, task_res in results.get("results", {}).items():
            # lm-eval v0.4+ uses "acc,none" / "acc_norm,none" keys
            acc = (task_res.get("acc,none",
                   task_res.get("acc_norm,none",
                   task_res.get("acc",
                   task_res.get("acc_norm",
                   task_res.get("perplexity,none",
                   task_res.get("perplexity", "N/A")))))))
            print(f"    {task_name}: {acc}")
        return results
    except Exception as e:
        print(f"  Downstream eval failed: {e}")
        return {}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CEDL 100M Benchmark")
    parser.add_argument("--model", type=str, default="CEDL",
                        help="Model tag or 'all'")
    parser.add_argument("--dataset", type=str, default="wikitext103")
    parser.add_argument("--tpu", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--verify-params", action="store_true",
                        help="Print param counts and exit")
    parser.add_argument("--max-steps", type=int, default=30_000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.verify_params:
        verify_all_params()
        return

    cfg = Config(
        dataset=args.dataset,
        tpu=args.tpu,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Performance: allow TF32 for matmuls on Ampere+ GPUs
    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    # Device setup
    if cfg.tpu:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        print(f"Using TPU: {device}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU (will be slow!)")

    # Data: tokenize once, rebuild loaders per model with different batch sizes
    train_loader, val_loader, test_loader, tokenizer = load_data(cfg)
    # Cache the datasets for fast loader rebuilding
    _train_ds = train_loader.dataset
    _val_ds = val_loader.dataset
    _test_ds = test_loader.dataset

    # Models to run
    models_to_run = ALL_MODELS if args.model == "all" else [args.model]

    results_table = {}

    for tag in models_to_run:
        # CEDL needs smaller batch (feedback doubles memory); others can use more
        if tag == "CEDL":
            cfg.batch_size = min(args.batch_size, 4)
        else:
            cfg.batch_size = args.batch_size
        train_loader, val_loader, test_loader = make_loaders(
            _train_ds, _val_ds, _test_ds, cfg.batch_size, cfg.tpu)

        print(f"\n{'='*60}")
        print(f"  Model: {tag} (batch_size={cfg.batch_size})")
        print(f"{'='*60}")

        model = build_model(tag, cfg.vocab_size, cfg.max_seq).to(device)
        n_params = count_params(model)
        print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

        # torch.compile for ~1.5-2x speedup (skip stateful models)
        if device.type == 'cuda' and not args.eval_only and tag not in ("Transformer-XL", "CEDL", "Mamba"):
            try:
                model = torch.compile(model)
                print("  torch.compile: enabled")
            except Exception as e:
                print(f"  torch.compile: skipped ({e})")

        if args.eval_only:
            # Auto-find checkpoint: use --checkpoint if given, else look for best.pt
            ckpt = args.checkpoint or os.path.join(cfg.save_dir, f"{tag}_best.pt")
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location='cpu', weights_only=True)
                # Strip _orig_mod. prefix from torch.compile'd checkpoints
                if any(k.startswith('_orig_mod.') for k in state):
                    state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
                model.load_state_dict(state)
                print(f"  Loaded checkpoint: {ckpt}")
            else:
                print(f"  WARNING: no checkpoint found at {ckpt}, skipping {tag}")
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        else:
            best_val_ppl = train(model, tag, cfg, device, train_loader, val_loader)
            # Load best checkpoint for test eval
            best_path = os.path.join(cfg.save_dir, f"{tag}_best.pt")
            if os.path.exists(best_path):
                model.load_state_dict(torch.load(best_path, map_location='cpu', weights_only=True))

        # Ensure CEDL feedback is fully enabled for evaluation
        raw_m = model._orig_mod if hasattr(model, '_orig_mod') else model
        if isinstance(raw_m, CEDLTwoLoop100M):
            raw_m.feedback_alpha.fill_(1.0)

        # Test perplexity
        test_ppl = evaluate(model, test_loader, device, max_steps=500, tpu=cfg.tpu)
        print(f"\n  Test Perplexity: {test_ppl:.2f}")

        # Structured benchmark skipped at 100M (PPL + downstream are the metrics)
        struct_results = {}

        # Downstream eval
        downstream = run_downstream_eval(model, tag, device)

        results_table[tag] = {
            "params": n_params,
            "test_ppl": test_ppl,
            "structured": struct_results,
            "downstream": downstream,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*90}")
    print(f"{'Model':20s} {'Params':>10s} {'Test PPL':>10s} {'Struct A':>9s} {'Struct B':>9s} {'Struct C':>9s} {'Overall':>9s}")
    print("-" * 90)
    for tag, res in results_table.items():
        sr = res.get("structured", {})
        a_vals = [v for k, d in sr.items() if k.startswith("A") for v in d.values()]
        b_vals = [v for k, d in sr.items() if k.startswith("B") for v in d.values()]
        c_vals = [v for k, d in sr.items() if k.startswith("C") for v in d.values()]
        a_avg = sum(a_vals) / len(a_vals) if a_vals else 0.0
        b_avg = sum(b_vals) / len(b_vals) if b_vals else 0.0
        c_avg = sum(c_vals) / len(c_vals) if c_vals else 0.0
        all_vals = a_vals + b_vals + c_vals
        overall = sum(all_vals) / len(all_vals) if all_vals else 0.0
        print(f"{tag:20s} {res['params']/1e6:>8.1f}M {res['test_ppl']:>10.2f} {a_avg:>8.1%} {b_avg:>8.1%} {c_avg:>8.1%} {overall:>8.1%}")


if __name__ == "__main__":
    main()

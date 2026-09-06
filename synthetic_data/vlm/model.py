#!/usr/bin/env python3
"""A small decoder-only VLM trained from scratch.

    frozen CLIP embedding (512-d)
        -> Projector (MLP)            -> n_prefix visual tokens of width d_model
        -> [visual tokens] [BOS] question [SEP] answer [EOS]
        -> causal Transformer decoder -> next-char logits

Loss is next-token cross-entropy on the answer tokens and the final EOS only; the
visual prefix and the question are context. Nothing here is pretrained.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import PAD_ID, EOS_ID, IGNORE


class Projector(nn.Module):
    """Maps one pooled CLIP vector to ``n_prefix`` tokens of width ``d_model``."""

    def __init__(self, in_dim, d_model, n_prefix=4, hidden=None, dropout=0.0):
        super().__init__()
        hidden = hidden or 4 * d_model
        self.n_prefix = n_prefix
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_prefix * d_model),
        )

    def forward(self, feats):
        return self.net(feats).view(-1, self.n_prefix, self.d_model)


class MoEFFN(nn.Module):
    """Top-1 mixture-of-experts feed-forward sublayer (Switch-Transformer style).

    Only the FFN is replicated; attention, embeddings, the projector and the
    LayerNorms stay shared. With top-1 routing each token therefore activates
    exactly one expert FFN, so the *active* parameter count equals the dense
    model's while total parameters grow by (n_experts - 1) FFNs per layer.

    The chosen expert's output is scaled by its router probability, which is what
    gives the router a gradient at all -- with a hard argmax and no scaling the
    routing decision would be non-differentiable and the router would never learn.
    """

    def __init__(self, d_model, n_experts=2, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(),
                          nn.Linear(4 * d_model, d_model))
            for _ in range(n_experts)])
        self.last_aux = None
        self.last_frac = None
        self.last_top = None      # (B, T) top-1 expert index, for diagnostics

    def forward(self, x, valid=None):
        B, T, C = x.shape
        logits = self.router(x)
        probs = F.softmax(logits.float(), dim=-1)
        top = probs.argmax(dim=-1)                       # (B, T)
        gate = probs.gather(-1, top[..., None]).squeeze(-1).to(x.dtype)

        # Dense-compute-and-mask: simple and exact. It does not save FLOPs at this
        # scale, but the parameters each token actually uses are one expert's.
        y = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            m = top == i
            if m.any():
                y = y + torch.where(m[..., None], expert(x), torch.zeros_like(x))
        y = y * gate[..., None]
        self.last_top = top.detach()

        # Switch load-balancing auxiliary loss over real (non-pad) positions:
        # E * sum_i f_i * P_i, minimised at 1.0 when the load is uniform. Without
        # it top-1 routing collapses onto a single expert.
        if valid is None:
            valid = torch.ones(B, T, dtype=torch.bool, device=x.device)
        v = valid.reshape(-1)
        if v.any():
            flat_top = top.reshape(-1)[v]
            flat_p = probs.reshape(-1, self.n_experts)[v]
            f = torch.zeros(self.n_experts, device=x.device)
            f.scatter_add_(0, flat_top, torch.ones_like(flat_top, dtype=f.dtype))
            f = f / f.sum()
            P = flat_p.mean(0)
            self.last_aux = self.n_experts * (f * P).sum()
            self.last_frac = f.detach()
        return y


class Block(nn.Module):
    """Pre-norm transformer block with causal self-attention."""

    def __init__(self, d_model, n_heads, dropout=0.0, n_experts=1, causal=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.causal = causal
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.n_experts = n_experts
        if n_experts > 1:
            self.mlp = MoEFFN(d_model, n_experts, dropout)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(d_model, 4 * d_model), nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = dropout

    def forward(self, x, valid=None, key_padding_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        shape = (B, T, self.n_heads, self.head_dim)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        attn_mask = None
        if not self.causal and key_padding_mask is not None:
            # (B, 1, 1, T) True = attend. Bidirectional attention would otherwise
            # read padded positions, which a causal mask made impossible.
            attn_mask = key_padding_mask[:, None, None, :]
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=self.causal,
            dropout_p=self.attn_dropout if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.drop(self.proj(y))
        h = self.ln2(x)
        h = self.mlp(h, valid) if self.n_experts > 1 else self.mlp(h)
        return x + self.drop(h)


class TinyVLM(nn.Module):
    def __init__(self, vocab_size, feat_dim=512, d_model=128, n_layers=4,
                 n_heads=4, n_prefix=4, max_len=192, dropout=0.1, n_experts=1):
        super().__init__()
        self.n_prefix = n_prefix
        self.max_len = max_len
        self.n_experts = n_experts
        self.projector = Projector(feat_dim, d_model, n_prefix, dropout=dropout)
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len + n_prefix, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout, n_experts) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight          # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_active_params(self):
        """Parameters a single token actually uses (top-1 routing)."""
        total = self.n_params()
        if self.n_experts <= 1:
            return total
        per_expert = sum(p.numel() for p in self.blocks[0].mlp.experts[0].parameters())
        idle = per_expert * (self.n_experts - 1) * len(self.blocks)
        return total - idle

    def aux_loss(self):
        """Mean Switch load-balancing loss over layers (1.0 == perfectly uniform)."""
        vals = [b.mlp.last_aux for b in self.blocks
                if self.n_experts > 1 and b.mlp.last_aux is not None]
        if not vals:
            return None
        return torch.stack(vals).mean()

    def expert_fractions(self):
        return [b.mlp.last_frac for b in self.blocks if self.n_experts > 1]

    def backbone(self, feats, ids):
        """Hidden states over [visual prefix | text]; shape (B, n_prefix + T, C)."""
        vis = self.projector(feats)
        txt = self.tok_emb(ids)
        h = torch.cat([vis, txt], dim=1)
        T = h.size(1)
        assert T <= self.pos_emb.num_embeddings, \
            f'sequence {T} exceeds max {self.pos_emb.num_embeddings}'
        h = self.drop(h + self.pos_emb(torch.arange(T, device=h.device))[None])
        valid = None
        if self.n_experts > 1:
            # visual prefix is always real; text positions only where not padding
            txt_valid = ids != PAD_ID
            valid = torch.cat([torch.ones(ids.size(0), self.n_prefix,
                                          dtype=torch.bool, device=h.device),
                               txt_valid], dim=1)
        for blk in self.blocks:
            h = blk(h, valid)
        return self.ln_f(h)

    def forward(self, feats, ids, labels=None):
        h = self.backbone(feats, ids)
        # Drop the visual prefix and the final text position: hidden state at text
        # index t predicts text token t+1.
        logits = self.head(h[:, self.n_prefix:-1])
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
                ignore_index=IGNORE)
        return logits, loss

    @torch.no_grad()
    def generate(self, feats, prompt_ids, max_new_tokens=64):
        """Greedy decode. ``prompt_ids`` is (B, P) with no padding (equal lengths)."""
        self.eval()
        ids = prompt_ids
        done = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device)
        out = [[] for _ in range(ids.size(0))]
        for _ in range(max_new_tokens):
            h = self.backbone(feats, ids[:, -(self.max_len):])
            nxt = self.head(h[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD_ID), nxt)
            for i, t in enumerate(nxt.tolist()):
                if not done[i] and t != EOS_ID:
                    out[i].append(t)
            done = done | (nxt == EOS_ID)
            if bool(done.all()):
                break
            ids = torch.cat([ids, nxt[:, None]], dim=1)
        return out

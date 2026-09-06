#!/usr/bin/env python3
"""Bidirectional transformer for autoregressive generation cast as discrete flow matching.

The answer is a fixed-length slot sequence of N positions. A *state* at timestep t
has the first t answer positions revealed and the remaining N-t held at [MASK]:

    t=0   [M][M][M][M] ...
    t=1   [a][M][M][M] ...
    t=2   [a][b][M][M] ...

The model sees one such state plus the conditioning (visual prefix + question) and
the scalar t, attends **bidirectionally** over everything, and emits a distribution
at every answer position. Advancing t -> t+1 reveals position t, so the transition
is deterministic everywhere except at position t, whose value is exactly what the
model must predict. That makes left-to-right AR a particular unmasking schedule of
a discrete flow / masked-diffusion model, and lets the two be compared directly.

Two differences from the causal decoder in model.py:

* **No causal mask.** Every position attends to every other, so a revealed token at
  the far right (if the schedule ever put one there) would inform position t. Under
  the strict left-to-right schedule nothing sits to the right of t, so the extra
  capacity is only used to look back -- which is what makes this a clean A/B against
  the causal model rather than a different task.
* **Timestep conditioning.** t is passed explicitly. Under the left-to-right schedule
  t is recoverable from the mask pattern, so this is redundant; it is kept because a
  general DFM schedule (random unmasking order) makes t genuinely informative, and
  keeping it means the schedule can be swapped without touching the model.

The [MASK] symbol is appended to the end of the tokenizer's index space, so existing
tokenizer files and checkpoints keep their indices.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Block, Projector


class BidirectionalDFM(nn.Module):
    def __init__(self, vocab_size, feat_dim=512, d_model=128, n_layers=4, n_heads=4,
                 n_prefix=4, max_prompt=64, answer_len=56, dropout=0.1):
        super().__init__()
        self.n_prefix = n_prefix
        self.answer_len = answer_len
        self.max_prompt = max_prompt
        self.mask_id = vocab_size            # [MASK] appended past the real vocab
        self.vocab_size = vocab_size

        self.projector = Projector(feat_dim, d_model, n_prefix, dropout=dropout)
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)   # +1 for [MASK]
        self.pos_emb = nn.Embedding(n_prefix + max_prompt + answer_len, d_model)
        self.time_emb = nn.Embedding(answer_len + 1, d_model)  # t = 0 .. answer_len
        self.seg_emb = nn.Embedding(3, d_model)                # prefix / prompt / answer
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, dropout, causal=False) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        # Not tied to tok_emb: the embedding table has vocab+1 rows ([MASK]) while
        # the head must never place mass on [MASK], so the shapes differ.
        self.head = nn.Linear(d_model, vocab_size, bias=False)
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

    def make_state(self, answer, t):
        """State x_t: answer positions < t revealed, >= t masked. t is (B,) or scalar."""
        B, N = answer.shape
        if not torch.is_tensor(t):
            t = torch.full((B,), int(t), device=answer.device, dtype=torch.long)
        pos = torch.arange(N, device=answer.device)[None, :]
        revealed = pos < t[:, None]
        return torch.where(revealed, answer, torch.full_like(answer, self.mask_id))

    def forward(self, feats, prompt_ids, prompt_mask, state, t):
        """Distribution at every answer position, given the state at timestep t.

        feats       (B, feat_dim)   frozen CLIP embedding
        prompt_ids  (B, P)          question tokens, right-padded
        prompt_mask (B, P)          True where the prompt is real
        state       (B, N)          answer slots, [MASK] where not yet revealed
        t           (B,) or int     timestep

        returns     (B, N, vocab)   logits; softmax gives p(x_{t+1} | x_t) at each slot
        """
        B, P = prompt_ids.shape
        N = state.shape[1]
        if not torch.is_tensor(t):
            t = torch.full((B,), int(t), device=state.device, dtype=torch.long)

        vis = self.projector(feats)                                    # (B, K, C)
        prm = self.tok_emb(prompt_ids)                                 # (B, P, C)
        ans = self.tok_emb(state)                                      # (B, N, C)
        h = torch.cat([vis, prm, ans], dim=1)
        T = h.shape[1]

        seg = torch.cat([
            torch.zeros(self.n_prefix, dtype=torch.long, device=h.device),
            torch.ones(P, dtype=torch.long, device=h.device),
            torch.full((N,), 2, dtype=torch.long, device=h.device)])
        h = h + self.pos_emb(torch.arange(T, device=h.device))[None] + self.seg_emb(seg)[None]
        # timestep conditioning, broadcast over the answer slots only
        h[:, self.n_prefix + P:] = h[:, self.n_prefix + P:] + self.time_emb(t)[:, None, :]
        h = self.drop(h)

        keep = torch.cat([
            torch.ones(B, self.n_prefix, dtype=torch.bool, device=h.device),
            prompt_mask,
            torch.ones(B, N, dtype=torch.bool, device=h.device)], dim=1)
        for blk in self.blocks:
            h = blk(h, key_padding_mask=keep)
        h = self.ln_f(h[:, self.n_prefix + P:])                        # answer slots
        return self.head(h)

    @torch.no_grad()
    def generate(self, feats, prompt_ids, prompt_mask, eos_id, pad_id):
        """Left-to-right unmasking: N steps, revealing position t at step t."""
        self.eval()
        B = feats.shape[0]
        N = self.answer_len
        state = torch.full((B, N), self.mask_id, dtype=torch.long, device=feats.device)
        done = torch.zeros(B, dtype=torch.bool, device=feats.device)
        for t in range(N):
            logits = self(feats, prompt_ids, prompt_mask, state, t)
            nxt = logits[:, t].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
            state[:, t] = nxt
            done = done | (nxt == eos_id)
            if bool(done.all()):
                state[:, t + 1:] = pad_id
                break
        return state

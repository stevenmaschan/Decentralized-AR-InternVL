#!/usr/bin/env python3
"""Full-model MoE: two complete VLMs and a learned router, trained jointly.

Contrast with the layerwise MoE in model.py, which replicates only the FFN
sublayer and routes every *token* independently. Here each expert is an entire
TinyVLM -- its own attention, embeddings, projector and FFNs -- and the router
makes a single decision per *sample*, because a whole model has to process the
whole sequence. That makes it the joint-training analogue of the hand-split
decentralised experts: same 2x parameters and same 1x inference cost, but the
partition is learned instead of given.

Loss (--routing sparse, the default). Hard top-1: the router picks one expert per
sample, only that expert runs, and the objective is

    L = NLL_top(x) - log p_top(x)

i.e. -log[ p_top(x) * p_expert_top(y | x) ], the Viterbi / hard-EM approximation
to the mixture likelihood. The -log p_top term is what gives the router its
gradient, and it pushes p_top *up* for the expert that was chosen.

Note this is NOT the same as scaling the loss by the gate, L = p_top * NLL_top.
Switch-Transformer scales the expert's *output* (y = p_i * E_i(x)), where driving
p down corrupts the representation and raises the loss. Transplanting that scaling
onto the loss inverts the incentive: since p_top >= 1/E, the router can minimise
p_top * NLL_top simply by staying maximally uncertain. An earlier version of this
file did exactly that and the router collapsed to p_top = 0.5000 on every sample.

Loss (--routing soft). The exact mixture likelihood
-logsumexp_i(log p_i - NLL_i), i.e. classical Jacobs-style MoE. It optimises a
quantity inference never gets: both experts contribute to the loss, but at
decode time only the argmax expert runs. Kept behind a flag so the soft-vs-sparse
difference can be measured rather than assumed.

Under sparse routing one expert runs per sample in both regimes, so active
parameters equal one dense model plus the router.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import IGNORE, PAD_ID
from .model import TinyVLM


class SampleRouter(nn.Module):
    """One routing decision per sample, from the image *and* the question.

    Giving the router the question text matters: this sweep showed that splits
    determined by the question (task type) tie or beat dense, while splits
    determined by the image (alpha) lose badly. A router that saw only CLIP
    features could not express the good partition at all.

    It is kept deliberately small (~21k parameters, +1.5% on top of one expert).
    A full-model router must encode the question, so it cannot be as near-free as
    the layerwise router's 1k; but telling the handful of question templates apart
    is an easy problem and does not need capacity.
    """

    def __init__(self, feat_dim, vocab_size, n_experts=2, d_text=32, d_hidden=32,
                 jitter=0.01):
        super().__init__()
        self.jitter = jitter
        self.q_emb = nn.Embedding(vocab_size, d_text)
        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim + d_text),
            nn.Linear(feat_dim + d_text, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, n_experts),
        )

    def forward(self, feats, prompt_ids, prompt_mask):
        m = prompt_mask.unsqueeze(-1).float()
        q = (self.q_emb(prompt_ids) * m).sum(1) / m.sum(1).clamp(min=1.0)
        x = torch.cat([feats, q], dim=-1)
        if self.training and self.jitter > 0:
            # Switch-Transformer multiplicative jitter: perturbs the router input
            # so top-1 selection explores instead of locking in whichever expert
            # it happened to prefer at initialisation.
            x = x * torch.empty_like(x).uniform_(1.0 - self.jitter,
                                                 1.0 + self.jitter)
        return self.net(x)


class FullModelMoE(nn.Module):
    def __init__(self, vocab_size, feat_dim=512, n_experts=2, routing='sparse',
                 jitter=0.01, **expert_kwargs):
        super().__init__()
        assert routing in ('sparse', 'soft')
        self.n_experts = n_experts
        self.routing = routing
        self.experts = nn.ModuleList([
            TinyVLM(vocab_size, feat_dim=feat_dim, **expert_kwargs)
            for _ in range(n_experts)])
        self.router = SampleRouter(feat_dim, vocab_size, n_experts, jitter=jitter)
        self.last_frac = None
        self.last_aux = None

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def n_active_params(self):
        """One expert plus the router runs per sample."""
        per = sum(p.numel() for p in self.experts[0].parameters())
        r = sum(p.numel() for p in self.router.parameters())
        return per + r

    @staticmethod
    def _per_sample_nll(logits, labels):
        """Sum of token NLL per sample (answer tokens only)."""
        tgt = labels[:, 1:]
        ll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                             tgt.reshape(-1), ignore_index=IGNORE,
                             reduction='none').view(tgt.shape)
        return ll.sum(dim=1)

    def forward(self, feats, ids, labels):
        prompt_mask = (labels == IGNORE) & (ids != PAD_ID)
        logits_r = self.router(feats, ids, prompt_mask)
        logp = F.log_softmax(logits_r.float(), dim=-1)          # (B, E)
        probs = logp.exp()
        top = probs.argmax(dim=1)                                # (B,)
        n_answer_tokens = (labels[:, 1:] != IGNORE).sum().clamp(min=1)

        if self.routing == 'sparse':
            # Hard top-1: run each expert only on the samples routed to it, so the
            # forward pass is one expert per sample -- the same computation as
            # inference. Scale by p_top to give the router its gradient.
            nll_top = torch.zeros(feats.size(0), device=feats.device,
                                  dtype=torch.float32)
            for i in range(self.n_experts):
                m = top == i
                if not bool(m.any()):
                    continue
                out = self.experts[i](feats[m], ids[m])[0]
                nll_top[m] = self._per_sample_nll(out, labels[m])
            # -log[p_top * p_expert(y|x)] = NLL_top - log p_top
            logp_top = logp.gather(1, top[:, None]).squeeze(1)
            loss = (nll_top - logp_top).sum() / n_answer_tokens
            with torch.no_grad():
                self.routed_ce = nll_top.sum() / n_answer_tokens
                self.mixture_nll = self.routed_ce
        else:
            nlls = torch.stack(
                [self._per_sample_nll(self.experts[i](feats, ids)[0], labels)
                 for i in range(self.n_experts)], dim=1)          # (B, E)
            loss = (-torch.logsumexp(logp - nlls, dim=1)).sum() / n_answer_tokens
            with torch.no_grad():
                self.mixture_nll = loss.detach()
                self.routed_ce = (nlls.gather(1, top[:, None]).squeeze(1).sum()
                                  / n_answer_tokens)

        # Switch-style load balance over samples, so one expert cannot take all
        f = torch.zeros(self.n_experts, device=feats.device)
        f.scatter_add_(0, top, torch.ones_like(top, dtype=f.dtype))
        f = f / f.sum().clamp(min=1)
        self.last_aux = self.n_experts * (f * probs.mean(0)).sum()
        self.last_frac = f.detach()
        return loss

    @torch.no_grad()
    def route(self, feats, ids, prompt_mask):
        return self.router(feats, ids, prompt_mask).argmax(dim=-1)

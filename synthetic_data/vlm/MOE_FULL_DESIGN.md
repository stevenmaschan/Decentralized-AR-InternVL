# Full-model hard MoE — design notes

Two complete VLMs and a learned router that picks one **per sample**. Contrast
with the layerwise MoE, which replicates only the FFN sublayer and routes every
**token**, independently, in every block.

Code: `moe_full.py` (model), `train_moe_full.py` (training / routed decoding).

## Routing

A whole model must process the whole sequence, so routing granularity is the
sample, not the token — one decision, made before anything is generated.

The router is a small MLP over `[CLIP features (512), mean-pooled character
embedding of the question (32)]` -> 2 logits, ~21k parameters (+1.6% on top of
one expert). It must see the **question**, not just the image: in this project
splits determined by the question (task type) tie or beat dense, while splits
determined by the image (alpha) lose ~2 points. A router on CLIP features alone
could not express the good partition at all.

Selection is `argmax`. The batch is then split and each expert runs only on the
samples routed to it, so the forward pass is one expert per sample — the same
computation as inference. Active parameters equal one dense model plus the router.

## Loss

    L = NLL_top(x) - log p_top(x)   =   -log[ p_top(x) * p_expert_top(y | x) ]

This is the Viterbi / hard-EM approximation to the mixture likelihood
`p(y|x) = sum_i p_i(x) * p_i(y|x)`. The `-log p_top` term is what gives the router
a gradient — a bare argmax is non-differentiable — and it pushes confidence *up*
on the expert that was chosen.

**What not to do.** Switch-Transformer scales the expert's *output*
(`y = p_i * E_i(x)`), where driving `p` down corrupts the representation and
raises the cross-entropy. That trick does not transplant onto a whole model: you
cannot scale output logits by `p` without distorting the predicted distribution.
An earlier version of this file scaled the *loss* instead, `L = p_top * NLL_top`.
Since `p_top >= 1/E`, that objective is minimised by staying maximally uncertain,
and the router collapsed to `p_top = 0.5000` on 100% of validation samples —
routing became a coin flip decided by numerical noise. Confidence after the fix:
mean 0.9956, 98.5% of samples above 0.90.

## Stabilisers

* **Switch load-balancing auxiliary loss**, `E * sum_i f_i * P_i`, coefficient
  0.01, computed over samples. Minimised at 1.0 under uniform load. Without it,
  competitive top-1 training has a rich-get-richer failure mode.
* **Multiplicative jitter** on the router input, `x * U[0.99, 1.01]`, train only.
  Top-1 selection is otherwise deterministic from initialisation, so whichever
  expert the router happens to prefer early gets reinforced with no exploration.

## Deliberate omissions

No capacity factor / token dropping (with 2 experts and no cap, nothing is ever
dropped), and no router z-loss. Both are standard in large Switch/ST-MoE setups
and would matter at more experts.

## Diagnostics worth logging

* **Router confidence** `p_top`. If it sits at `1/E`, the objective is wrong —
  this is the check that caught the bug above.
* **Task separation**: share of tokens routed to expert 0, split by question
  family. 0pp means the router ignores the input; the working version reached
  ~62pp at 100k.
* **Load balance** per epoch. Oscillation between extremes signals too weak an
  aux coefficient for the sample-level signal (a batch of 128 gives 128 routing
  decisions, versus ~24k for a token-level router — a 190x weaker signal).

## Outcome on this task

The router does find the task split, and it still loses: -1.78 vs dense at 150k
(-1.64 at 100k), against +0.30 for the same two-model capacity given the task
split by hand. Specialisation is real but costs each expert half its data, and
the router's 60-80% purity does not recover that. The layerwise MoE, with 190x
more routing decisions per step, ties dense instead (+0.14).

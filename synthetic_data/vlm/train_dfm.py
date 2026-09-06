#!/usr/bin/env python3
"""Train the bidirectional DFM model: AR generation as a left-to-right unmasking flow.

    python -m synthetic_data.vlm.train_dfm --out-dir .../dfm --annotations ... --features ...

Training. For each example a timestep t ~ Uniform{0..N-1} is drawn, the state x_t is
built (first t answer slots revealed, rest [MASK]), and the model predicts the clean
answer. Two losses:

  --loss step   cross-entropy at position t only. This is exactly the AR factorisation
                p(x_t | x_<t, cond): with t uniform the expected gradient covers every
                position, and the model is trained on precisely the quantity generation
                uses. Directly comparable to the causal decoder.
  --loss all    cross-entropy at every still-masked position (>= t). The usual
                masked-diffusion / x1-prediction objective: a denser signal per forward
                pass, but it also trains predictions the left-to-right sampler never
                queries.

Generation is N forward passes, revealing one position per step -- N times the compute
of causal AR decoding, which caches. The point of the exercise is the parameterisation,
not the speed.
"""

import json
import math
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .data import (CharTokenizer, FeatureStats, load_records, get_split,
                   PAD_ID, BOS_ID, SEP_ID, EOS_ID)
from .dfm_model import BidirectionalDFM
from .train import score, fmt_report


class DFMDataset(Dataset):
    """Fixed-length answer slots; the prompt is conditioning and is never masked."""

    def __init__(self, records, feats, tok, stats, answer_len, max_prompt):
        self.records, self.feats, self.tok, self.stats = records, feats, tok, stats
        self.answer_len, self.max_prompt = answer_len, max_prompt

    def __len__(self):
        return len(self.records)

    def prompt_ids(self, r):
        return ([BOS_ID] + self.tok.encode(r['question']) + [SEP_ID])[:self.max_prompt]

    def __getitem__(self, i):
        r = self.records[i]
        p = self.prompt_ids(r)
        a = self.tok.encode(r['answer']) + [EOS_ID]
        a = (a + [PAD_ID] * self.answer_len)[:self.answer_len]
        return {'feat': torch.from_numpy(np.ascontiguousarray(
                    self.stats.apply(self.feats[r['row']]))).float(),
                'prompt': torch.tensor(p, dtype=torch.long),
                'answer': torch.tensor(a, dtype=torch.long),
                'index': i}


def collate(batch, max_prompt):
    P = max(len(b['prompt']) for b in batch)
    ids = torch.full((len(batch), P), PAD_ID, dtype=torch.long)
    msk = torch.zeros((len(batch), P), dtype=torch.bool)
    for i, b in enumerate(batch):
        L = len(b['prompt'])
        ids[i, :L] = b['prompt']; msk[i, :L] = True
    return {'feat': torch.stack([b['feat'] for b in batch]),
            'prompt': ids, 'prompt_mask': msk,
            'answer': torch.stack([b['answer'] for b in batch]),
            'index': torch.tensor([b['index'] for b in batch])}


@torch.no_grad()
def generate_all(model, ds, device, batch_size=128):
    model.eval()
    out = [None] * len(ds)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0,
                        collate_fn=lambda b: collate(b, ds.max_prompt))
    pos = 0
    for b in loader:
        st = model.generate(b['feat'].to(device), b['prompt'].to(device),
                            b['prompt_mask'].to(device), EOS_ID, PAD_ID)
        for row in st.cpu().tolist():
            s = []
            for tkn in row:
                if tkn in (EOS_ID, PAD_ID):
                    break
                s.append(tkn)
            out[pos] = ds.tok.decode(s); pos += 1
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--annotations', required=True)
    ap.add_argument('--features', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--loss', choices=('step', 'all'), default='step')
    ap.add_argument('--t-mode', choices=('random', 'all'), default='random',
                    help="'random': one t ~ U{0..N-1} per example (cheap, unbiased); "
                         "'all': every t for every example, so one pass over the data "
                         "supervises all N prefixes -- the same coverage a causal model "
                         "gets for free, at N times the compute")
    ap.add_argument('--t-chunk', type=int, default=512,
                    help='sequences per forward in --t-mode all (memory bound)')
    ap.add_argument('--d-model', type=int, default=128)
    ap.add_argument('--n-layers', type=int, default=4)
    ap.add_argument('--n-heads', type=int, default=4)
    ap.add_argument('--n-prefix', type=int, default=4)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--eval-every', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.01)
    ap.add_argument('--warmup-frac', type=float, default=0.03)
    ap.add_argument('--grad-clip', type=float, default=1.0)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    records, feats = load_records(args.annotations, args.features)
    splits = {k: get_split(records, k) for k in ('train', 'val', 'test')}
    if args.max_train:
        splits['train'] = [r for r in splits['train'] if r['image_id'] < args.max_train]
    tok = (CharTokenizer.load(args.tokenizer) if args.tokenizer else
           CharTokenizer.from_texts([r['question'] for r in records]
                                    + [r['answer'] for r in records]))
    tok.save(out_dir / 'tokenizer.json')
    stats = FeatureStats.fit(feats, [r['row'] for r in splits['train']])

    answer_len = max(len(r['answer']) for r in records) + 1          # + EOS
    max_prompt = max(len(r['question']) for r in records) + 2        # + BOS, SEP
    print(f'train {len(splits["train"])}  val {len(splits["val"])}  '
          f'answer_len {answer_len}  max_prompt {max_prompt}  '
          f'loss={args.loss}  t-mode={args.t_mode}')
    if args.t_mode == 'all':
        print(f'  --t-mode all: {answer_len} timesteps per example -> '
              f'{args.batch_size * answer_len} sequences per update '
              f'(chunks of {args.t_chunk})')

    ds = {k: DFMDataset(v, feats, tok, stats, answer_len, max_prompt)
          for k, v in splits.items()}
    model = BidirectionalDFM(len(tok), feats.shape[1], args.d_model, args.n_layers,
                             args.n_heads, args.n_prefix, max_prompt, answer_len,
                             args.dropout).to(device)
    print(f'bidirectional DFM: {model.n_params()/1e6:.2f}M parameters, '
          f'[MASK] id {model.mask_id}')

    decay = [p for _, p in model.named_parameters() if p.ndim >= 2]
    nodecay = [p for _, p in model.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': args.weight_decay},
                             {'params': nodecay, 'weight_decay': 0.0}],
                            lr=args.lr, betas=(0.9, 0.95))
    loader = DataLoader(ds['train'], batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0,
                        collate_fn=lambda b: collate(b, max_prompt))
    total = max(1, len(loader)) * args.epochs
    warm = max(1, int(args.warmup_frac * total))

    def lr_at(s):
        if s < warm:
            return args.lr * s / warm
        pr = (s - warm) / max(1, total - warm)
        return args.lr * 0.5 * (1 + math.cos(math.pi * min(1.0, pr)))

    best = {'val_em': -1.0, 'epoch': -1}; hist = []; step = 0; t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train(); run = cnt = 0.0
        for b in loader:
            for g in opt.param_groups:
                g['lr'] = lr_at(step)
            feat = b['feat'].to(device, non_blocking=True)
            prm = b['prompt'].to(device, non_blocking=True)
            pmk = b['prompt_mask'].to(device, non_blocking=True)
            ans = b['answer'].to(device, non_blocking=True)
            B, N = ans.shape
            opt.zero_grad(set_to_none=True)

            if args.t_mode == 'random':
                pairs = [(feat, prm, pmk, ans,
                          torch.randint(0, N, (B,), device=device))]
                denom = 1
            else:
                # every example x every timestep; chunked so memory stays bounded,
                # gradients accumulated so the update matches one full expansion
                fe = feat.repeat_interleave(N, 0); pe = prm.repeat_interleave(N, 0)
                me = pmk.repeat_interleave(N, 0); ae = ans.repeat_interleave(N, 0)
                te = torch.arange(N, device=device).repeat(B)
                pairs = [(fe[i:i + args.t_chunk], pe[i:i + args.t_chunk],
                          me[i:i + args.t_chunk], ae[i:i + args.t_chunk],
                          te[i:i + args.t_chunk])
                         for i in range(0, fe.shape[0], args.t_chunk)]
                denom = len(pairs)

            tot_loss = 0.0; tot_k = 0
            for f_, p_, m_, a_, t_ in pairs:
                state = model.make_state(a_, t_)
                logits = model(f_, p_, m_, state, t_)
                if args.loss == 'step':
                    tgt = a_.gather(1, t_[:, None]).squeeze(1)
                    pred = logits.gather(
                        1, t_[:, None, None].expand(-1, 1, logits.size(-1))).squeeze(1)
                    loss = F.cross_entropy(pred.float(), tgt)
                    k = a_.shape[0]
                else:
                    pos = torch.arange(N, device=device)[None, :]
                    masked = pos >= t_[:, None]
                    loss = F.cross_entropy(logits.float()[masked], a_[masked])
                    k = int(masked.sum())
                (loss / denom).backward()
                tot_loss += loss.item() * k; tot_k += k

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            run += tot_loss; cnt += tot_k; step += 1

        e = {'epoch': epoch, 'train_loss': run / max(1, cnt)}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            rep = score(splits['val'], generate_all(model, ds['val'], device))
            e['val_exact_match'] = rep['exact_match']
            print(f'epoch {epoch:>3}  loss {e["train_loss"]:.4f}  '
                  f'val EM {100*rep["exact_match"]:.2f}%  ({time.time()-t0:.0f}s)')
            if rep['exact_match'] > best['val_em']:
                best = {'val_em': rep['exact_match'], 'epoch': epoch}
                torch.save({'model': model.state_dict(), 'args': vars(args),
                            'stats': stats.state_dict(), 'vocab': tok.itos,
                            'answer_len': answer_len, 'max_prompt': max_prompt},
                           out_dir / 'best.pt')
        else:
            print(f'epoch {epoch:>3}  loss {e["train_loss"]:.4f}  '
                  f'({time.time()-t0:.0f}s)')
        hist.append(e)

    ck = out_dir / 'best.pt'
    if ck.exists():
        model.load_state_dict(torch.load(ck, map_location=device)['model'])
        print(f'\nloaded best (epoch {best["epoch"]}, val EM {100*best["val_em"]:.2f}%)')
    res = {'args': vars(args), 'history': hist, 'best': best}
    for sp in ('val', 'test'):
        res[sp] = score(splits[sp], generate_all(model, ds[sp], device))
        print('\n' + fmt_report(sp.upper(), res[sp]))
    (out_dir / 'results.json').write_text(json.dumps(res, indent=2))
    print(f'\nWrote {out_dir / "results.json"}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Data plumbing for the tiny VLM: char tokenizer, splits, dataset, collation.

The vision side is a *precomputed, frozen* CLIP embedding per image (one 512-d
vector from clustering/extract_clip_features.py). A learned projector turns it
into visual prefix tokens; nothing in the CLIP tower is trained.

Text is tokenized at character level. The corpus is tiny and closed (a handful of
question templates; answers are letter sequences, small integers, and
comma-separated shape names), so a ~70-symbol char vocabulary trains from scratch
without importing anyone else's tokenizer -- and it lets the model spell OCR
answers it has never seen as a whole string.
"""

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

PAD, BOS, SEP, EOS = '<pad>', '<bos>', '<sep>', '<eos>'
SPECIALS = [PAD, BOS, SEP, EOS]
PAD_ID, BOS_ID, SEP_ID, EOS_ID = 0, 1, 2, 3
IGNORE = -100


class CharTokenizer:
    """Character-level tokenizer over a closed vocabulary."""

    def __init__(self, chars):
        self.itos = list(SPECIALS) + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    @classmethod
    def from_texts(cls, texts):
        return cls(sorted({c for t in texts for c in t}))

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return ''.join(self.itos[i] for i in ids if i >= len(SPECIALS))

    def save(self, path):
        Path(path).write_text(json.dumps({'itos': self.itos}))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())['itos'][len(SPECIALS):])


def load_records(annotations, features, keys=None):
    """Read annotations.jsonl and align each QA pair with its CLIP feature row."""
    feat_path = Path(features)
    keys_path = Path(keys) if keys else \
        feat_path.with_name(feat_path.stem + '_keys.json')
    feats = np.load(feat_path).astype(np.float32)
    row_of = {int(k): i for i, k in enumerate(json.load(open(keys_path)))}

    records = []
    for line in open(annotations):
        a = json.loads(line)
        img = a['image_id']
        if img not in row_of:                      # image without features
            continue
        conv = a['conversations']
        question = conv[0]['value'].split('\n', 1)[-1]
        records.append({
            'image_id': img,
            'row': row_of[img],
            'question': question,
            'answer': conv[1]['value'],
            'task_id': a['task_id'],
            'task_type': a['task_type'],
            'alpha': a['alpha'],
            'split': a.get('split'),
        })
    return records, feats


def split_by_image(records, val_frac=0.1, test_frac=0.1, seed=0):
    """Assign each image to train/val/test by a stable hash of its image_id.

    Hashing rather than permuting the ids *present in this file* matters: an
    expert shard holds only a subset of the images, and a permutation over that
    subset would place images differently than the dense run does. With a hash,
    every subset inherits the same assignment, so a dense model and any shard
    trained from the same corpus share their val/test images exactly and their
    scores are comparable.
    """
    # A dataset generated with --val-count/--test-count carries its split
    # explicitly; use it verbatim so every shard agrees and the sizes are exact.
    if records and records[0].get('split'):
        out = {'train': [], 'val': [], 'test': []}
        for r in records:
            out[r['split']].append(r)
        return out

    def unit(image_id):
        h = hashlib.md5(f'{seed}:{image_id}'.encode()).digest()
        return int.from_bytes(h[:8], 'big') / 2 ** 64

    # Test is carved from a fixed window at the bottom of the hash range and val
    # from the window just above it, so changing --val-frac grows val into train
    # and leaves the test set untouched. Runs with different val sizes stay
    # comparable on test.
    out = {'train': [], 'val': [], 'test': []}
    for r in records:
        x = unit(r['image_id'])
        if x < test_frac:
            out['test'].append(r)
        elif x < test_frac + val_frac:
            out['val'].append(r)
        else:
            out['train'].append(r)
    return out


def get_split(records, name):
    """Fetch one split, or a '+'-joined union of splits (e.g. 'val+test').

    Combining val and test doubles the evaluation set without retraining. Note
    that val was used for checkpoint selection while test was not, so a combined
    figure is slightly optimistic relative to test alone -- report both when the
    difference matters.
    """
    parts = split_by_image(records)
    out = []
    for key in name.split('+'):
        if key not in parts:
            raise KeyError(f'unknown split {key!r}')
        out.extend(parts[key])
    return out


class FeatureStats:
    """Standardization fitted on the training rows only."""

    def __init__(self, mean, std, mode='standardize'):
        self.mean, self.std, self.mode = mean, std, mode

    @classmethod
    def fit(cls, feats, rows, mode='standardize'):
        X = feats[sorted(set(rows))]
        return cls(X.mean(0), np.clip(X.std(0), 1e-6, None), mode)

    def apply(self, X):
        if self.mode == 'none':
            return X
        if self.mode == 'l2':
            return X / np.clip(np.linalg.norm(X, axis=-1, keepdims=True), 1e-8, None)
        return (X - self.mean) / self.std

    def state_dict(self):
        # Plain lists, not ndarrays, so the checkpoint stays loadable under
        # torch>=2.6's default weights_only=True.
        return {'mean': self.mean.tolist(), 'std': self.std.tolist(),
                'mode': self.mode}

    @classmethod
    def from_state(cls, d):
        return cls(np.asarray(d['mean'], dtype=np.float32),
                   np.asarray(d['std'], dtype=np.float32), d['mode'])


class VQADataset(Dataset):
    """Yields the token sequence [BOS] question [SEP] answer [EOS] plus its features.

    Only the answer tokens and the final EOS carry loss; the question is context.
    """

    def __init__(self, records, feats, tokenizer, stats, no_vision=False):
        self.records = records
        self.feats = feats
        self.tok = tokenizer
        self.stats = stats
        self.no_vision = no_vision

    def __len__(self):
        return len(self.records)

    def prompt_ids(self, rec):
        return [BOS_ID] + self.tok.encode(rec['question']) + [SEP_ID]

    def __getitem__(self, i):
        r = self.records[i]
        prompt = self.prompt_ids(r)
        answer = self.tok.encode(r['answer']) + [EOS_ID]
        ids = prompt + answer
        labels = [IGNORE] * len(prompt) + answer
        x = np.zeros_like(self.feats[0]) if self.no_vision \
            else self.stats.apply(self.feats[r['row']])
        return {
            'feat': torch.from_numpy(np.ascontiguousarray(x)).float(),
            'ids': torch.tensor(ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'index': i,
        }


def collate(batch):
    n = max(len(b['ids']) for b in batch)
    ids = torch.full((len(batch), n), PAD_ID, dtype=torch.long)
    labels = torch.full((len(batch), n), IGNORE, dtype=torch.long)
    for i, b in enumerate(batch):
        L = len(b['ids'])
        ids[i, :L] = b['ids']
        labels[i, :L] = b['labels']
    return {
        'feat': torch.stack([b['feat'] for b in batch]),
        'ids': ids,
        'labels': labels,
        'index': torch.tensor([b['index'] for b in batch]),
    }

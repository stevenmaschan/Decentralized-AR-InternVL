#!/usr/bin/env python3
"""
Generate a synthetic image+text dataset for small-scale VLM experiments.

Every image contains a set of geometric shapes and (usually) a short random
letter sequence (3-6 letters by default) rendered as text. Two families of tasks are derived from the ground truth used to render the
image:

    OCR              - read the letter sequence written on the image
    shape reasoning  - count / name shapes by type, color and number of angles

By default one question is sampled per image: the OCR question with probability
``alpha``, otherwise a uniformly chosen shape question. Pass
``--question-sampling all`` for the full fixed battery on every image.

Each image is parametrised by a single scalar ``alpha`` in [0, 1] that trades off
text prominence against shape prominence:

    alpha = 0  ->  small, faint text          + large, opaque shapes
    alpha = 1  ->  large, opaque text         + small, faint shapes

so ``alpha`` interpolates the image between "shape-reasoning friendly" and
"OCR friendly". It is drawn from Beta(b, b) via ``--alpha-beta`` (default 0.5,
U-shaped, so most images sit near one extreme; pass 1.0 for uniform). Shapes are laid out on a jittered grid so they never overlap,
which keeps the counting answers unambiguous.

Outputs (under --output-dir):

    images/000000.png ...              rendered images
    metadata.jsonl                     one line per image: full ground truth
    annotations.jsonl                  one line per QA pair, InternVL chat format
    preview.png                        sanity-check grid of the first N images

Usage:
    python synthetic_data/generate_shapes_dataset.py \
        --output-dir data/synthetic_shapes --num-samples 1000 --seed 0
"""

import os
import json
import math
import random
import argparse
import multiprocessing as mp
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #

SHAPE_TYPES = ['circle', 'triangle', 'ellipse', 'square', 'rectangle', 'hexagon']

SHAPE_PLURAL = {
    'circle': 'circles',
    'triangle': 'triangles',
    'ellipse': 'ellipses',
    'square': 'squares',
    'rectangle': 'rectangles',
    'hexagon': 'hexagons',
}

# Number of corners/angles of each shape type (round shapes have none).
SHAPE_ANGLES = {
    'circle': 0,
    'ellipse': 0,
    'triangle': 3,
    'square': 4,
    'rectangle': 4,
    'hexagon': 6,
}

# No shape has 1 or 2 corners, so ">= 1" and ">= 3" would select exactly the same
# set (every polygon) and ask the same question twice under different wording.
ANGLE_THRESHOLDS = [3, 4, 6]

COLORS = {
    'red':    (219, 50, 45),
    'green':  (46, 160, 67),
    'blue':   (37, 99, 235),
    'yellow': (238, 197, 40),
    'orange': (243, 130, 32),
    'purple': (142, 68, 190),
    'pink':   (240, 118, 178),
    'brown':  (140, 90, 50),
    'gray':   (128, 128, 128),
    'black':  (28, 28, 30),
}
COLOR_NAMES = list(COLORS)

# Dark, high-contrast ink colors for the text so that legibility is controlled by
# the alpha parameter rather than by an unlucky color draw.
TEXT_COLORS = {
    'black':      (20, 20, 22),
    'dark blue':  (20, 40, 120),
    'dark red':   (130, 25, 25),
    'dark green': (20, 85, 45),
    'dark gray':  (70, 70, 75),
    'navy':       (18, 30, 70),
}

FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
    '/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSerif.ttf',
    '/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf',
    '/usr/share/fonts/truetype/freefont/FreeMono.ttf',
    '/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf',
]

LETTERS = 'abcdefghijklmnopqrstuvwxyz'


def sample_text(rng, min_len, max_len):
    """A random letter sequence of length in [min_len, max_len].

    Deliberately not dictionary words: a small vocabulary of real words repeats
    across the dataset and CLIP resolves word identity almost perfectly, so the
    feature space clusters by word rather than by the alpha parameter.
    """
    n = rng.randint(min_len, max_len)
    style = rng.random()
    if style < 0.45:                       # all lowercase
        return ''.join(rng.choice(LETTERS) for _ in range(n))
    if style < 0.80:                       # ALL UPPERCASE
        return ''.join(rng.choice(LETTERS) for _ in range(n)).upper()
    return ''.join(                        # MiXeD case
        rng.choice(LETTERS).upper() if rng.random() < 0.5 else rng.choice(LETTERS)
        for _ in range(n))

# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #


def _regular_polygon(cx, cy, r, n_sides, rotation_deg, start_deg=-90.0):
    """Vertices of a regular n-gon with circumradius ``r``."""
    pts = []
    for i in range(n_sides):
        a = math.radians(start_deg + rotation_deg + 360.0 * i / n_sides)
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _rotate(points, cx, cy, rotation_deg):
    a = math.radians(rotation_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for x, y in points:
        dx, dy = x - cx, y - cy
        out.append((cx + dx * ca - dy * sa, cy + dx * sa + dy * ca))
    return out


def _ellipse_points(cx, cy, rx, ry, rotation_deg, n=72):
    pts = [(cx + rx * math.cos(2 * math.pi * i / n),
            cy + ry * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return _rotate(pts, cx, cy, rotation_deg)


def shape_polygon(shape_type, cx, cy, r, rotation_deg, aspect):
    """Polygon approximation of ``shape_type`` inscribed in a circle of radius r."""
    if shape_type == 'circle':
        return _ellipse_points(cx, cy, r, r, 0.0)
    if shape_type == 'ellipse':
        return _ellipse_points(cx, cy, r, r * aspect, rotation_deg)
    if shape_type == 'triangle':
        return _regular_polygon(cx, cy, r, 3, rotation_deg)
    if shape_type == 'square':
        return _regular_polygon(cx, cy, r, 4, rotation_deg, start_deg=-45.0)
    if shape_type == 'hexagon':
        return _regular_polygon(cx, cy, r, 6, rotation_deg)
    if shape_type == 'rectangle':
        # Half-diagonal is r, so the rectangle stays inside the placement circle.
        hw = r / math.sqrt(1.0 + aspect * aspect)
        hh = hw * aspect
        pts = [(cx - hw, cy - hh), (cx + hw, cy - hh),
               (cx + hw, cy + hh), (cx - hw, cy + hh)]
        return _rotate(pts, cx, cy, rotation_deg)
    raise ValueError(f'unknown shape type: {shape_type}')


def poly_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def boxes_overlap(a, b, pad=0.0):
    return not (a[2] + pad < b[0] or b[2] + pad < a[0] or
                a[3] + pad < b[1] or b[3] + pad < a[1])


def lerp(a, b, t):
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# Sampling of the scene
# --------------------------------------------------------------------------- #


def sample_alpha(rng, cfg):
    """Draw the image parameter on [0, 1].

    Two families, selected by --alpha-dist:

    ``beta``    alpha ~ Beta(b, b). b = 1 is uniform; b < 1 is U-shaped, pushing
                mass towards both ends (b = 0.5 is the arcsine distribution).

    ``expmix``  an equal mixture of two truncated exponentials, one decaying from
                0 and one decaying from 1 -- density proportional to
                e^(-r*alpha) + e^(-r*(1-alpha)). Also symmetric and U-shaped, but
                with exponential rather than power-law tails, so the mass piles up
                against the endpoints instead of diverging at them. r = 1 is very
                nearly uniform; larger r concentrates on the ends.
    """
    if cfg['alpha_dist'] == 'expmix':
        r = cfg['alpha_rate']
        u = rng.random()
        # inverse CDF of Exp(r) truncated to [0, 1]
        x = -math.log1p(-u * (1.0 - math.exp(-r))) / r
        return x if rng.random() < 0.5 else 1.0 - x
    beta = cfg['alpha_beta']
    if beta == 1.0:
        return rng.random()
    return rng.betavariate(beta, beta)


def sample_scene(rng, alpha, cfg):
    """Sample the ground-truth description of one image."""
    n_shapes = rng.randint(cfg['min_shapes'], cfg['max_shapes'])

    # Grid with a few more cells than shapes -> varied positions, no overlap.
    grid = max(2, math.ceil(math.sqrt(n_shapes * 1.6)))
    cells = rng.sample([(i, j) for i in range(grid) for j in range(grid)], n_shapes)
    cell = cfg['image_size'] / grid
    cell_half = cell / 2.0

    # alpha=0 -> big shapes, alpha=1 -> small shapes (fraction of the cell).
    size_lo = lerp(cfg['shape_size_lo'][0], cfg['shape_size_lo'][1], alpha)
    size_hi = lerp(cfg['shape_size_hi'][0], cfg['shape_size_hi'][1], alpha)
    # alpha=0 -> opaque shapes, alpha=1 -> faint shapes.
    op_lo = lerp(cfg['shape_opacity_lo'][0], cfg['shape_opacity_lo'][1], alpha)
    op_hi = lerp(cfg['shape_opacity_hi'][0], cfg['shape_opacity_hi'][1], alpha)

    shapes = []
    for (gi, gj) in cells:
        shape_type = rng.choice(SHAPE_TYPES)
        color_name = rng.choice(COLOR_NAMES)
        r = cell_half * rng.uniform(size_lo, size_hi)
        r = max(r, cfg['min_radius'])

        jitter = max(0.0, cell_half - r - cfg['cell_pad'])
        cx = (gi + 0.5) * cell + rng.uniform(-jitter, jitter)
        cy = (gj + 0.5) * cell + rng.uniform(-jitter, jitter)

        if shape_type == 'ellipse':
            aspect = rng.uniform(0.35, 0.65)
        elif shape_type == 'rectangle':
            aspect = rng.choice([rng.uniform(0.30, 0.60), rng.uniform(1.7, 3.3)])
        else:
            aspect = 1.0

        rotation = 0.0 if shape_type == 'circle' else rng.uniform(0.0, 360.0)
        opacity = rng.uniform(op_lo, op_hi)

        pts = shape_polygon(shape_type, cx, cy, r, rotation, aspect)
        shapes.append({
            'type': shape_type,
            'color': color_name,
            'center': [round(cx, 2), round(cy, 2)],
            'radius': round(r, 2),
            'rotation': round(rotation, 2),
            'aspect': round(aspect, 3),
            'opacity': round(opacity, 3),
            'bbox': [round(v, 2) for v in poly_bbox(pts)],
            'n_angles': SHAPE_ANGLES[shape_type],
            '_points': pts,
        })

    text = None
    if rng.random() > cfg['p_no_text']:
        word = sample_text(rng, cfg['text_min_len'], cfg['text_max_len'])
        text = {
            'string': word,
            'font': rng.choice(cfg['fonts']),
            # alpha=0 -> small, faint text; alpha=1 -> large, opaque text.
            'size': int(round(lerp(rng.uniform(*cfg['text_size_lo']),
                                   rng.uniform(*cfg['text_size_hi']), alpha))),
            'opacity': round(lerp(rng.uniform(*cfg['text_opacity_lo']),
                                  rng.uniform(*cfg['text_opacity_hi']), alpha), 3),
            'rotation': round(rng.uniform(-cfg['text_max_rotation'],
                                          cfg['text_max_rotation']), 2),
            'color': rng.choice(list(TEXT_COLORS)),
        }

    return {'alpha': round(alpha, 4), 'shapes': shapes, 'text': text}


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_scene(scene, cfg, rng):
    """Render a sampled scene; returns (PIL.Image, text_bbox_or_None)."""
    size = cfg['image_size']
    ss = cfg['supersample']
    big = size * ss

    bg = tuple(rng.randint(*cfg['bg_range']) for _ in range(3))
    canvas = Image.new('RGB', (big, big), bg)

    # Shapes never overlap, so a single RGBA layer composited once gives exactly
    # the per-shape opacity requested.
    layer = Image.new('RGBA', (big, big), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for sh in scene['shapes']:
        rgb = COLORS[sh['color']]
        a = int(round(sh['opacity'] * 255))
        pts = [(x * ss, y * ss) for x, y in sh['_points']]
        outline = tuple(int(c * 0.65) for c in rgb) + (min(255, a + 35),)
        draw.polygon(pts, fill=rgb + (a,), outline=outline, width=max(1, ss))
    canvas = Image.alpha_composite(canvas.convert('RGBA'), layer)

    text_bbox = None
    if scene['text'] is not None:
        t = scene['text']
        font = ImageFont.truetype(t['font'], t['size'] * ss)
        probe = ImageDraw.Draw(Image.new('RGBA', (1, 1)))
        l, top, r, b = probe.textbbox((0, 0), t['string'], font=font)
        tw, th = max(1, r - l), max(1, b - top)

        pad = 6 * ss
        tile = Image.new('RGBA', (tw + 2 * pad, th + 2 * pad), (0, 0, 0, 0))
        ImageDraw.Draw(tile).text(
            (pad - l, pad - top), t['string'], font=font,
            fill=TEXT_COLORS[t['color']] + (int(round(t['opacity'] * 255)),))
        tile = tile.rotate(t['rotation'], resample=Image.BICUBIC, expand=True)

        tile_w, tile_h = tile.size
        max_x, max_y = big - tile_w, big - tile_h
        if max_x <= 0 or max_y <= 0:
            # Too large for the canvas: shrink it to fit rather than clipping it.
            scale = min(big / tile_w, big / tile_h) * 0.95
            tile = tile.resize((max(1, int(tile_w * scale)), max(1, int(tile_h * scale))),
                               Image.LANCZOS)
            tile_w, tile_h = tile.size
            max_x, max_y = max(0, big - tile_w), max(0, big - tile_h)

        shape_boxes = [sh['bbox'] for sh in scene['shapes']]
        px = py = 0
        for attempt in range(cfg['text_placement_tries'] + 1):
            px = rng.randint(0, max_x) if max_x > 0 else 0
            py = rng.randint(0, max_y) if max_y > 0 else 0
            box = [px / ss, py / ss, (px + tile_w) / ss, (py + tile_h) / ss]
            # Last attempt: accept whatever we have rather than loop forever.
            if attempt == cfg['text_placement_tries']:
                break
            if not any(boxes_overlap(box, sb) for sb in shape_boxes):
                break
        canvas.alpha_composite(tile, (px, py))
        text_bbox = [round(px / ss, 2), round(py / ss, 2),
                     round((px + tile_w) / ss, 2), round((py + tile_h) / ss, 2)]

    return canvas.convert('RGB').resize((size, size), Image.LANCZOS), text_bbox


# --------------------------------------------------------------------------- #
# Question / answer construction
# --------------------------------------------------------------------------- #


def count_maps(shapes):
    type_counts = {t: 0 for t in SHAPE_TYPES}
    color_counts = {c: 0 for c in COLOR_NAMES}
    for sh in shapes:
        type_counts[sh['type']] += 1
        color_counts[sh['color']] += 1
    return type_counts, color_counts


def _pick_target(rng, present, absent, p_absent):
    """One query target, occasionally something absent from the image (answer 0)."""
    if absent and (not present or rng.random() < p_absent):
        return rng.choice(absent)
    return rng.choice(present)


# --- one builder per task ---------------------------------------------------- #

def q_ocr(scene, rng, cfg, tc, cc):
    text = scene['text']['string'] if scene['text'] else cfg['no_text_answer']
    return (1, 'ocr', 'Recognise text on the image.', text)


def q_type(scene, rng, cfg, tc, cc):
    present = [t for t in SHAPE_TYPES if tc[t] > 0]
    absent = [t for t in SHAPE_TYPES if tc[t] == 0]
    t = _pick_target(rng, present, absent, cfg['p_absent_query'])
    return (2, 'shape_reasoning',
            f'How many {SHAPE_PLURAL[t]} are on the image?', str(tc[t]))


def q_color(scene, rng, cfg, tc, cc):
    present = [c for c in COLOR_NAMES if cc[c] > 0]
    absent = [c for c in COLOR_NAMES if cc[c] == 0]
    c = _pick_target(rng, present, absent, cfg['p_absent_query'])
    return (3, 'shape_reasoning',
            f'How many shapes of {c} color are on the image?', str(cc[c]))


def q_names(scene, rng, cfg, tc, cc):
    return (4, 'shape_reasoning', 'Name types of shapes you see on the image.',
            ', '.join(t for t in SHAPE_TYPES if tc[t] > 0))


def q_angles(scene, rng, cfg, tc, cc):
    n = rng.choice(ANGLE_THRESHOLDS)
    count = sum(1 for sh in scene['shapes'] if sh['n_angles'] >= n)
    return (5, 'shape_reasoning',
            f'How many shapes have at least {n} angles?', str(count))


# task_id -> builder, so tasks can be switched off by id via --exclude-tasks
QUESTION_BUILDERS = {1: q_ocr, 2: q_type, 3: q_color, 4: q_names, 5: q_angles}
SHAPE_TASK_IDS = [2, 3, 4, 5]


def enabled_shape_questions(cfg):
    return [QUESTION_BUILDERS[t] for t in SHAPE_TASK_IDS
            if t not in cfg['exclude_tasks']]


def build_qa(scene, rng, cfg):
    """Build the list of (task_id, task_type, question, answer) for one scene.

    Two modes (``--question-sampling``):

    ``alpha``  -- sample ``--questions-per-image`` questions. Each one is the OCR
        question with probability ``alpha`` and otherwise a uniformly chosen shape
        question. Since alpha ~ Beta(b, b), the per-image OCR probability is itself
        Beta-distributed across the dataset: text-prominent images get asked about
        text, shape-prominent images about shapes.

    ``all``    -- emit the full fixed battery (1 OCR + 2 type + 2 color + 1 names
        + 2 angle by default), i.e. every image carries the same task mix. Useful
        for evaluation, where you want full coverage rather than a sample.
    """
    tc, cc = count_maps(scene['shapes'])
    shape_qs = enabled_shape_questions(cfg)
    ocr_on = 1 not in cfg['exclude_tasks']

    if cfg['question_sampling'] == 'alpha':
        qa = []
        for _ in range(cfg['questions_per_image']):
            # If one side is switched off entirely, always draw from the other.
            want_ocr = (ocr_on and not shape_qs) or (
                ocr_on and rng.random() < scene['alpha'])
            if want_ocr:
                qa.append(q_ocr(scene, rng, cfg, tc, cc))
            else:
                qa.append(rng.choice(shape_qs)(scene, rng, cfg, tc, cc))
        return qa, tc, cc

    # --- exhaustive mode --------------------------------------------------- #
    present_types = [t for t in SHAPE_TYPES if tc[t] > 0]
    absent_types = [t for t in SHAPE_TYPES if tc[t] == 0]
    present_colors = [c for c in COLOR_NAMES if cc[c] > 0]
    absent_colors = [c for c in COLOR_NAMES if cc[c] == 0]

    qa = [q_ocr(scene, rng, cfg, tc, cc)] if ocr_on else []

    def pick(present, absent, k):
        """k distinct-where-possible targets, occasionally absent ones."""
        out = []
        pool = list(present)
        rng.shuffle(pool)
        for _ in range(k):
            if absent and (not pool or rng.random() < cfg['p_absent_query']):
                out.append(rng.choice(absent))
            elif pool:
                out.append(pool.pop())
            elif present:
                out.append(rng.choice(present))
        return out

    if 2 not in cfg['exclude_tasks']:
        for t in pick(present_types, absent_types, cfg['n_type_questions']):
            qa.append((2, 'shape_reasoning',
                       f'How many {SHAPE_PLURAL[t]} are on the image?', str(tc[t])))
    if 3 not in cfg['exclude_tasks']:
        for c in pick(present_colors, absent_colors, cfg['n_color_questions']):
            qa.append((3, 'shape_reasoning',
                       f'How many shapes of {c} color are on the image?', str(cc[c])))
    if 4 not in cfg['exclude_tasks']:
        qa.append(q_names(scene, rng, cfg, tc, cc))
    if 5 not in cfg['exclude_tasks']:
        for n in rng.sample(ANGLE_THRESHOLDS, k=min(cfg['n_angle_questions'],
                                                    len(ANGLE_THRESHOLDS))):
            count = sum(1 for sh in scene['shapes'] if sh['n_angles'] >= n)
            qa.append((5, 'shape_reasoning',
                       f'How many shapes have at least {n} angles?', str(count)))

    return qa, tc, cc


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def build_config(args):
    fonts = [f for f in FONT_CANDIDATES if os.path.exists(f)]
    if not fonts:
        raise RuntimeError('no TrueType fonts found; install fonts-dejavu / fonts-freefont-ttf')

    s = args.image_size / 448.0  # font/geometry constants are tuned for 448 px
    return {
        'image_size': args.image_size,
        'supersample': args.supersample,
        'fonts': fonts,
        'min_shapes': args.min_shapes,
        'max_shapes': args.max_shapes,
        'min_radius': 9.0 * s,
        'cell_pad': 4.0 * s,
        'bg_range': (233, 253),
        # (value at alpha=0, value at alpha=1) as a fraction of the grid cell
        'shape_size_lo': (0.55, 0.18),
        'shape_size_hi': (0.95, 0.38),
        'shape_opacity_lo': (0.75, 0.12),
        'shape_opacity_hi': (1.00, 0.28),
        # text: (min, max) of the uniform draw at alpha=0 and at alpha=1
        'text_size_lo': (12 * s, 20 * s),
        'text_size_hi': (60 * s, 105 * s),
        'text_opacity_lo': (0.25, 0.45),
        'text_opacity_hi': (0.85, 1.00),
        'text_max_rotation': args.text_max_rotation,
        'text_placement_tries': 60,
        'p_no_text': args.p_no_text,
        'alpha_beta': args.alpha_beta,
        'alpha_dist': args.alpha_dist,
        'alpha_rate': args.alpha_rate,
        'question_sampling': args.question_sampling,
        'exclude_tasks': set(args.exclude_tasks),
        'questions_per_image': args.questions_per_image,
        'text_min_len': args.text_min_len,
        'text_max_len': args.text_max_len,
        'p_absent_query': args.p_absent_query,
        'no_text_answer': args.no_text_answer,
        'n_type_questions': args.n_type_questions,
        'n_color_questions': args.n_color_questions,
        'n_angle_questions': args.n_angle_questions,
    }


def make_preview(paths, out_path, cols=5, thumb=224):
    rows = math.ceil(len(paths) / cols)
    grid = Image.new('RGB', (cols * thumb, rows * thumb), (255, 255, 255))
    for i, p in enumerate(paths):
        im = Image.open(p).resize((thumb, thumb), Image.LANCZOS)
        grid.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    grid.save(out_path)



# --------------------------------------------------------------------------- #
# Per-image worker
# --------------------------------------------------------------------------- #

_WORKER = {}


def _init_worker(cfg, img_dir, image_size, seed):
    _WORKER.update(cfg=cfg, img_dir=Path(img_dir), image_size=image_size, seed=seed)


def render_one(idx):
    """Render image ``idx`` and return its metadata record.

    The RNG is seeded per image from a *string* (CPython hashes str/bytes seeds
    with SHA-512, unlike tuples, whose hash() is salted per process), so output
    is identical no matter how many worker processes run or in what order.
    """
    cfg, img_dir = _WORKER['cfg'], _WORKER['img_dir']
    rng = random.Random(f"{_WORKER['seed']}:{idx}")

    alpha = sample_alpha(rng, cfg)
    scene = sample_scene(rng, alpha, cfg)
    image, text_bbox = render_scene(scene, cfg, rng)
    image.save(img_dir / f'{idx:06d}.png')

    qa, type_counts, color_counts = build_qa(scene, rng, cfg)
    return {
        'image_id': idx,
        'image': f'images/{idx:06d}.png',
        'width': _WORKER['image_size'],
        'height': _WORKER['image_size'],
        'alpha': scene['alpha'],
        'text': (dict(scene['text'], bbox=text_bbox) if scene['text'] else None),
        'n_shapes': len(scene['shapes']),
        'shapes': [{k: v for k, v in sh.items() if k != '_points'}
                   for sh in scene['shapes']],
        'type_counts': type_counts,
        'color_counts': color_counts,
        'qa': [{'task_id': t, 'task_type': tt, 'question': q, 'answer': a}
               for t, tt, q, a in qa],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--output-dir', type=str, default='data/synthetic_shapes')
    ap.add_argument('--num-samples', type=int, default=1000)
    ap.add_argument('--image-size', type=int, default=448)
    ap.add_argument('--supersample', type=int, default=3,
                    help='anti-aliasing factor used while rendering')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-shapes', type=int, default=1)
    ap.add_argument('--max-shapes', type=int, default=10)
    ap.add_argument('--p-no-text', type=float, default=0.15,
                    help='fraction of images rendered without any text')
    ap.add_argument('--p-absent-query', type=float, default=0.25,
                    help='probability a counting question targets an absent type/color')
    ap.add_argument('--no-text-answer', type=str, default='',
                    help='answer for the OCR task when the image has no text '
                         '(default: the empty string, i.e. "produce nothing")')
    ap.add_argument('--question-sampling', choices=('alpha', 'all'), default='alpha',
                    help="'alpha': sample questions, OCR with probability alpha and "
                         "otherwise a uniform shape question; 'all': emit the full "
                         "fixed battery per image")
    ap.add_argument('--exclude-tasks', type=int, nargs='*', default=[],
                    metavar='TASK_ID',
                    help='task ids never to ask (1 ocr, 2 count-by-type, '
                         '3 count-by-color, 4 name-types, 5 count-by-angles)')
    ap.add_argument('--questions-per-image', type=int, default=1,
                    help="questions sampled per image in --question-sampling alpha")
    ap.add_argument('--n-type-questions', type=int, default=2)
    ap.add_argument('--n-color-questions', type=int, default=2)
    ap.add_argument('--n-angle-questions', type=int, default=2)
    ap.add_argument('--text-max-rotation', type=float, default=45.0)
    ap.add_argument('--alpha-dist', choices=('beta', 'expmix'), default='beta',
                    help="'beta': alpha ~ Beta(b, b); 'expmix': equal mixture of "
                         "two truncated exponentials decaying from 0 and from 1")
    ap.add_argument('--alpha-rate', type=float, default=1.0,
                    help='rate of the exponential components (--alpha-dist expmix); '
                         '1 is nearly uniform, larger concentrates on the ends')
    ap.add_argument('--alpha-beta', type=float, default=0.5,
                    help='alpha ~ Beta(b, b); b=1 is uniform, b<1 concentrates mass '
                         'at both ends (default 0.5 = arcsine)')
    ap.add_argument('--text-min-len', type=int, default=3,
                    help='minimum number of letters in the rendered sequence')
    ap.add_argument('--text-max-len', type=int, default=6,
                    help='maximum number of letters in the rendered sequence')
    ap.add_argument('--image-prefix', type=str, default='',
                    help='prefix prepended to the image path stored in annotations.jsonl')
    ap.add_argument('--dataset-name', type=str, default='synthetic_shapes',
                    help='key used in the emitted InternVL meta.json')
    ap.add_argument('--root', type=str, default=None,
                    help='"root" field of meta.json, i.e. the directory image paths are '
                         'resolved against (default: --output-dir)')
    ap.add_argument('--num-workers', type=int, default=0,
                    help='parallel rendering processes (0 = min(cpu_count, 32)); '
                         'output is identical for any worker count')
    ap.add_argument('--val-count', type=int, default=0,
                    help='images reserved for validation (taken from the end)')
    ap.add_argument('--test-count', type=int, default=0,
                    help='images reserved for test (taken from the very end)')
    ap.add_argument('--preview', type=int, default=25,
                    help='number of images in preview.png (0 to disable)')
    args = ap.parse_args()

    cfg = build_config(args)
    if not set(range(1, 6)) - cfg['exclude_tasks']:
        raise SystemExit('--exclude-tasks removed every task; nothing to generate')

    out_dir = Path(args.output_dir)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)

    n_workers = args.num_workers or min(os.cpu_count() or 1, 32)
    n_workers = max(1, min(n_workers, args.num_samples))
    print(f'Rendering {args.num_samples} images with {n_workers} worker(s)...')

    init_args = (cfg, str(img_dir), args.image_size, args.seed)
    if n_workers == 1:
        _init_worker(*init_args)
        records = map(render_one, range(args.num_samples))
        pool = None
    else:
        pool = mp.Pool(n_workers, initializer=_init_worker, initargs=init_args)
        # imap keeps results in index order while workers run ahead.
        records = pool.imap(render_one, range(args.num_samples), chunksize=16)

    meta_f = open(out_dir / 'metadata.jsonl', 'w')
    ann_f = open(out_dir / 'annotations.jsonl', 'w')

    # Explicit split by image index: train | val | test, contiguous from the
    # start. alpha and every other property is drawn i.i.d. per image, so index
    # ranges are unbiased, and any later subset (an expert shard) inherits the
    # assignment for free because it travels with the record.
    n_test, n_val = args.test_count, args.val_count
    n_train = args.num_samples - n_val - n_test
    if n_train <= 0:
        raise SystemExit('--val-count + --test-count leave no training images')

    def split_of(idx):
        if idx < n_train:
            return 'train'
        return 'val' if idx < n_train + n_val else 'test'

    if n_val or n_test:
        print(f'splits: train {n_train}, val {n_val}, test {n_test}')

    sample_id = 0
    stats = {'ocr': 0, 'shape_reasoning': 0, 'no_text': 0, 'shapes': 0}

    for idx, rec in enumerate(records):
        assert rec['image_id'] == idx, (rec['image_id'], idx)
        rec['split'] = split_of(idx)
        meta_f.write(json.dumps(rec) + '\n')

        for qa in rec['qa']:
            ann_f.write(json.dumps({
                'id': sample_id,
                'image_id': idx,
                'image': args.image_prefix + rec['image'],
                'width': rec['width'],
                'height': rec['height'],
                'alpha': rec['alpha'],
                'split': rec['split'],
                'task_id': qa['task_id'],
                'task_type': qa['task_type'],
                'conversations': [
                    {'from': 'human', 'value': f"<image>\n{qa['question']}"},
                    {'from': 'gpt', 'value': qa['answer']},
                ],
            }) + '\n')
            sample_id += 1
            stats[qa['task_type']] += 1

        stats['shapes'] += rec['n_shapes']
        if rec['text'] is None:
            stats['no_text'] += 1

        if (idx + 1) % 1000 == 0:
            print(f'  {idx + 1}/{args.num_samples} images', flush=True)

    if pool is not None:
        pool.close()
        pool.join()

    meta_f.close()
    ann_f.close()

    n_preview = min(args.preview, args.num_samples)
    if n_preview > 0:
        make_preview([img_dir / f'{i:06d}.png' for i in range(n_preview)],
                     out_dir / 'preview.png')

    # InternVL-style meta file, usable directly as --meta_path when training.
    root = args.root if args.root is not None else str(out_dir) + '/'
    with open(out_dir / 'meta.json', 'w') as f:
        json.dump({args.dataset_name: {
            'root': root,
            'annotation': str(out_dir / 'annotations.jsonl'),
            'data_augment': False,
            'repeat_time': 1,
            'length': sample_id,
        }}, f, indent=2)

    print(f'\nWrote {args.num_samples} images to {img_dir}')
    print(f'  {sample_id} QA pairs  ({stats["ocr"]} OCR, '
          f'{stats["shape_reasoning"]} shape reasoning)')
    print(f'  {stats["no_text"]} images without text, '
          f'{stats["shapes"] / max(1, args.num_samples):.2f} shapes/image on average')
    print(f'  annotations: {out_dir / "annotations.jsonl"}')
    print(f'  metadata:    {out_dir / "metadata.jsonl"}')
    print(f'  meta.json:   {out_dir / "meta.json"}')


if __name__ == '__main__':
    main()

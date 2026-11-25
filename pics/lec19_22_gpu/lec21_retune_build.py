"""Builds lec19_22_gpu/lec21_retune.ipynb. Cells are separated by lines '# %% [markdown]' / '# %%'."""
import json
import sys
from pathlib import Path

SRC = r'''
# %% [markdown]
# # Lecture 21: retune the SEGA and LEDITS++ examples
#
# Run this on the Hub GPU, a section at a time, and look at each grid before going on.
# Nothing is final until the **Save** cells at the end of each section; they write into
# `results-lec21/lec21/`, which goes back to the laptop.
#
# 1. **SEGA.** Find a street prompt and two noises (A and B) with parked cars, so that *− cars* has something
#    to remove; check that *+ autumn*, *+ snow*, *− cars* and *+ autumn − cars* all show; then save the full set
#    the demo needs (same settings as before: 50 steps, guidance 7.5, warm-up 10, momentum 0.3/0.6).
# 2. **LEDITS++.** Invert the cat photo, sweep edit scale × threshold (and inversion `skip`) for each edit,
#    pick the strongest clean result for each, and save them under the names the slide uses.
#
# Each sampling call takes about 2 s on the Hub GPU, so a 16-image grid is well under a minute.

# %%
# Setup. The bundle was written for diffusers 0.36.0, but any recent version should do. Only if a pipeline call
# below fails with an argument error, run   %pip install --user -q diffusers==0.36.0   and restart the kernel
# (a --user install affects every notebook on this Hub account, so remove it afterwards if the labs need theirs).
import os
os.environ.setdefault('L19_RESULTS', 'results-lec21')        # read by common.py: where the outputs go

import textwrap
import numpy as np
import torch
import diffusers
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont

from common import NEGATIVE, RESULTS, free, gen, gpu_info, inp, load_json, save_img, save_json, sd_pipe

print('diffusers', diffusers.__version__, '| torch', torch.__version__, '|', gpu_info())
if diffusers.__version__ != '0.36.0':
    print('(diffusers is not 0.36.0: fine unless a pipeline call fails; see the comment at the top of this cell)')
OUT = RESULTS / 'lec21'
info = load_json(OUT / 'info.json')
assert info and 'weight' in info, f'{OUT}/info.json is missing: unzip the whole bundle, not just the notebook'

try:
    FONT = ImageFont.load_default(size=15)
except TypeError:                                   # Pillow < 10.1
    FONT = ImageFont.load_default()


def grid(cells, ncol, size=230, title=None):
    """Show a labelled grid. cells: list of (PIL image or None, label)."""
    pad, lh = 6, 18
    nlines = max(len(textwrap.wrap(lab, 30)) for _, lab in cells) if cells else 1
    nrow = -(-len(cells) // ncol)
    th = lh + 4 if title else 0
    W, H = ncol * (size + pad) + pad, th + nrow * (size + nlines * lh + 2 * pad) + pad
    g = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(g)
    if title:
        d.text((pad, 2), title, fill='black', font=FONT)
    for i, (im, lab) in enumerate(cells):
        x = pad + (i % ncol) * (size + pad)
        y = th + pad + (i // ncol) * (size + nlines * lh + 2 * pad)
        if im is not None:
            g.paste(im.convert('RGB').resize((size, size)), (x, y))
        for j, line in enumerate(textwrap.wrap(lab, 30)):
            d.text((x, y + size + 2 + j * lh), line, fill='black', font=FONT)
    display(g)
    return g


# %% [markdown]
# ## 1. Semantic guidance (SEGA)
#
# ### 1a. Find a street with cars
#
# Before, the prompt was *"a photo of a quiet street in a small town, houses and trees"*, and seeds 21 and 22 gave
# streets with almost no cars. Two ways out, both tried below:
# * keep a prompt that doesn't mention cars and look through more seeds for ones that happen to have them;
# * mention parked cars in the prompt, so that *− cars* visibly removes what the prompt asked for.
#
# Edit `PROMPTS` and `SEEDS` and rerun until there are two good candidates: plenty of parked cars, trees that
# can turn orange, a recognisable street.

# %%
SEGA_STEPS, WARMUP, MOMENTUM = 50, 10, (0.3, 0.6)   # as in the deck's aside: keep these unless the deck changes

PROMPTS = [
    'a photo of a quiet street in a small town, houses and trees',
    'a photo of a street in a small town, houses and trees, cars parked along the road',
]
SEEDS = list(range(20, 28))

_sega = None


def sega_pipe():
    global _sega
    if _sega is None:
        from diffusers import SemanticStableDiffusionPipeline
        _sega = sd_pipe(SemanticStableDiffusionPipeline)
    return _sega


def sega_run(prompt, seed, edits=(), scales=(), threshold=0.95):
    """edits: [(text, reverse), ...], scales: one edit guidance scale per edit. Returns (image, map or None)."""
    pipe = sega_pipe()
    kw = {}
    if edits:
        kw = dict(editing_prompt=[e[0] for e in edits], reverse_editing_direction=[e[1] for e in edits],
                  edit_guidance_scale=list(scales), edit_threshold=[threshold] * len(edits),
                  edit_warmup_steps=[WARMUP] * len(edits), edit_momentum_scale=MOMENTUM[0],
                  edit_mom_beta=MOMENTUM[1])
    im = pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=SEGA_STEPS, guidance_scale=7.5,
              generator=gen(seed), **kw).images[0]
    m = None
    if edits and pipe.sem_guidance is not None:
        # |edit guidance| summed over the 4 latent channels: a 64 x 64 map per step of where the edit acts
        m = pipe.sem_guidance.float().abs().sum(dim=-3).squeeze().numpy().astype(np.float16)
    return im, m


for pi, p in enumerate(PROMPTS):
    grid([(sega_run(p, s)[0], f'prompt {pi}, seed {s}') for s in SEEDS], ncol=4, title=f'{pi}: {p}')

# %% [markdown]
# ### 1b. Try the edits on the candidates
#
# Put the chosen prompt and two seeds in `PROMPT` and `CANDIDATES` (more than two is fine, to compare). Each row
# shows no edit, then each edit at its scale. Edit `EDITS` (the text) and `SCALE` (one scale per edit) until all
# four show clearly at threshold 0.95. The demo's slider sweeps the autumn scale over 2–12, so autumn should
# be visible but not overdone at its `SCALE`. Keeping 6 for everything keeps the deck's text simplest.

# %%
PROMPT = PROMPTS[1]
CANDIDATES = [21, 22]
EDITS = {'autumn': ('autumn, orange leaves', False), 'snow': ('snow, winter', False),
         'cars': ('cars, parked cars', True)}         # True = push away from the concept
SCALE = {'autumn': 6, 'snow': 6, 'cars': 6}
COMBOS = [('autumn',), ('snow',), ('cars',), ('autumn', 'cars')]


def label(combo):
    return '  '.join(('− ' if EDITS[e][1] else '+ ') + f'"{EDITS[e][0]}" g{SCALE[e]}' for e in combo)


for s in CANDIDATES:
    cells = [(sega_run(PROMPT, s)[0], f'seed {s}, no edit')]
    cells += [(sega_run(PROMPT, s, [EDITS[e] for e in c], [SCALE[e] for e in c])[0], label(c)) for c in COMBOS]
    grid(cells, ncol=5, size=220)

# %% [markdown]
# Optional: if *− cars* is too weak or too strong, sweep its scale (and the threshold) on one seed.

# %%
SEED, EDIT = CANDIDATES[0], 'cars'
cells = [(sega_run(PROMPT, SEED, [EDITS[EDIT]], [g], th)[0], f'{EDIT} g{g} th{th}')
         for th in [0.95, 0.9] for g in [4, 6, 8, 10, 14]]
grid(cells, ncol=5, size=200)

# %% [markdown]
# ### 1c. Save the full SEGA set
#
# Set `SEEDS_AB` to the two chosen seeds (noise A, noise B), and check `PROMPT`, `EDITS` and `SCALE` above.
# This regenerates everything the two SEGA demos use (25 images, about a minute): no edit, the autumn strength
# sweep, the autumn threshold sweep on noise A, the other edits at their scale, and *where the edit acted* maps
# for noise A. The old SEGA files are deleted first, so nothing from seeds 21/22 lingers.

# %%
SEEDS_AB = [21, 22]
AUTUMN_SWEEP = [2, 4, 6, 8, 12]                     # the demo's strength slider
THRESHOLDS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99]       # the threshold demo's slider (autumn, at SCALE['autumn'])

def num(v):
    """6.0 -> '6', 7.5 -> '7.5': matches how the demo's JavaScript writes numbers into file names."""
    return str(int(v)) if float(v).is_integer() else str(v)


def gname(scales):
    """One scale if every edit uses the same one, else e.g. '6_8' (the demo builds the same string)."""
    return num(scales[0]) if len(set(scales)) == 1 else '_'.join(num(s) for s in scales)


d = OUT / 'sega'
if d.exists():
    for f in d.iterdir():
        f.unlink()
sweep = sorted(set(AUTUMN_SWEEP) | {SCALE['autumn']})
jobs = [((), [], 0.95, s) for s in SEEDS_AB]
jobs += [(('autumn',), [g], 0.95, s) for g in sweep for s in SEEDS_AB]
jobs += [(('autumn',), [SCALE['autumn']], th, SEEDS_AB[0]) for th in THRESHOLDS if th != 0.95]
jobs += [((e,), [SCALE[e]], 0.95, s) for e in ['snow', 'cars'] for s in SEEDS_AB]
jobs += [(('autumn', 'cars'), [SCALE['autumn'], SCALE['cars']], 0.95, s) for s in SEEDS_AB]
for edits, scales, th, seed in jobs:
    name = ('+'.join(edits) or 'none') + f'-g{gname(scales) if edits else 0}-th{th}-{seed}'
    im, m = sega_run(PROMPT, seed, [EDITS[e] for e in edits], scales, th)
    save_img(im, d / f'{name}.jpg')
    if m is not None and seed == SEEDS_AB[0]:
        np.save(d / f'maps-{name}.npy', m)
info['sega'] = {'prompt': PROMPT, 'edits': EDITS, 'steps': SEGA_STEPS, 'warmup': WARMUP,
                'momentum': list(MOMENTUM), 'seeds': SEEDS_AB, 'scale': SCALE,
                'autumn_sweep': sweep, 'thresholds': THRESHOLDS}
save_json(info, OUT / 'info.json')
files = sorted(d.glob('*.jpg'))
grid([(Image.open(f), f.stem) for f in files], ncol=6, size=170, title=f'{len(files)} SEGA images saved')

# %%
# Free the SEGA pipeline before LEDITS++ (the Hub GPU may be an 8 GB slice)
_sega = None
free()

# %% [markdown]
# ## 2. LEDITS++ on the cat photo
#
# Before, the edits used the diffusers defaults (edit scale 5, threshold 0.9) with `skip` 0.15 and no source
# prompt, and only *"a fox"* changed anything (the nose). The levers, roughly from gentle to strong:
# * **edit guidance scale** `g`: 5 is the default; 10–20 is usual for visible edits;
# * **threshold** `th`: lower lets the edit act over more of the image (0.9 → 0.6);
# * **skip**: the fraction of the noisiest steps left out of inversion and editing. Lower keeps less of the
#   photo's layout fixed and allows bigger changes;
# * **multi-concept edits**: e.g. *− "cat"* together with *+ "a fox"* (each edit below is a list of concepts).
#
# ### 2a. Invert, and check the reconstruction

# %%
from diffusers import LEditsPPPipelineStableDiffusion

INV_STEPS = 50
SOURCE_PROMPT = ''                  # try 'a photo of a cat' too: it can make concept edits on the subject easier
PHOTO = inp('cat.png')
lpipe = sd_pipe(LEditsPPPipelineStableDiffusion)
_inverted = None


def invert(skip, source=None):
    """Invert the photo (cached: re-inverting only when skip or the source prompt changes)."""
    global _inverted
    source = SOURCE_PROMPT if source is None else source
    if _inverted != (skip, source):
        lpipe.invert(image=PHOTO, source_prompt=source, num_inversion_steps=INV_STEPS, skip=skip,
                     generator=gen(0))
        _inverted = (skip, source)


def ledits(concepts, g, th, skip, source=None):
    """concepts: [(text, reverse), ...]; g and th apply to every concept. concepts=[] reconstructs."""
    invert(skip, source)
    kw = {}
    if concepts:
        n = len(concepts)
        kw = dict(editing_prompt=[c[0] for c in concepts], reverse_editing_direction=[c[1] for c in concepts],
                  edit_guidance_scale=[g] * n, edit_threshold=[th] * n)
    return lpipe(generator=gen(0), **kw).images[0]


grid([(PHOTO, 'the photo')] + [(ledits([], 0, 0, sk), f'no edit, skip {sk}') for sk in [0.15, 0.1, 0.05, 0.0]],
     ncol=5, size=220)

# %% [markdown]
# ### 2b. Sweep each edit
#
# For each edit: a grid of scale (columns) × threshold (rows) at one `skip`. Change `LEDITS`, `G`, `TH` and
# `SKIP` and rerun; to look at one edit only, set `ONLY = ['fox']`. Note the cell labels of the ones you like.

# %%
LEDITS = {
    'sleep': [('sleeping cat, closed eyes', False)],
    'fox': [('a fox', False)],
    'fox2': [('cat', True), ('a fox', False)],              # remove the cat and add a fox
    'paint': [('watercolour painting', False)],
    'stripes': [('stripes, tabby', True)],
}
G = [5, 10, 15, 20]
TH = [0.9, 0.75, 0.6]
SKIP = 0.15
ONLY = None                                                 # e.g. ['fox', 'fox2']

for key, concepts in LEDITS.items():
    if ONLY and key not in ONLY:
        continue
    desc = '  '.join(('− ' if r else '+ ') + f'"{t}"' for t, r in concepts)
    grid([(ledits(concepts, g, th, SKIP), f'{key} g{g} th{th} skip{SKIP}') for th in TH for g in G],
         ncol=len(G), size=200, title=f'{key}: {desc}')

# %% [markdown]
# Optional: the effect of `skip` (and the source prompt) on one edit at fixed scale and threshold.

# %%
KEY, g, th = 'fox', 15, 0.75
cells = [(ledits(LEDITS[KEY], g, th, sk, src), f'skip{sk} src "{src}"')
         for src in ['', 'a photo of a cat'] for sk in [0.15, 0.1, 0.05]]
grid(cells, ncol=3, size=220, title=f'{KEY} g{g} th{th}')

# %% [markdown]
# ### 2c. Save the chosen LEDITS++ edits
#
# One entry per image on the slide (the slide shows four edits next to *no edit*; keys become file names
# `ledits-<key>.jpg`). Each entry: the concepts, `g`, `th`, `skip` and source prompt. The reconstruction
# (`ledits-none.jpg`) uses the skip and source prompt of `NONE`. If the chosen edits use different `skip`s,
# say so when you bring the files back, since the slide text will need to mention it.

# %%
CHOSEN = {
    'sleep':   dict(concepts=LEDITS['sleep'],   g=15, th=0.75, skip=0.15, source=''),
    'fox':     dict(concepts=LEDITS['fox'],     g=15, th=0.75, skip=0.15, source=''),
    'paint':   dict(concepts=LEDITS['paint'],   g=15, th=0.75, skip=0.15, source=''),
    'stripes': dict(concepts=LEDITS['stripes'], g=15, th=0.75, skip=0.15, source=''),
}
NONE = dict(skip=0.15, source='')

inv = OUT / 'invert'
for f in inv.glob('ledits-*.jpg'):
    f.unlink()
cells = [(PHOTO, 'the photo')]
im = ledits([], 0, 0, NONE['skip'], NONE['source'])
save_img(im, inv / 'ledits-none.jpg')
cells.append((im, 'ledits-none'))
for key, c in CHOSEN.items():
    im = ledits(c['concepts'], c['g'], c['th'], c['skip'], c['source'])
    save_img(im, inv / f'ledits-{key}.jpg')
    cells.append((im, f"ledits-{key}: g{c['g']} th{c['th']} skip{c['skip']}"))
info['ledits'] = {'steps': INV_STEPS, 'scheduler': type(lpipe.scheduler).__name__, 'none': NONE,
                  **{k: {**c, 'concepts': [list(x) for x in c['concepts']]} for k, c in CHOSEN.items()}}
save_json(info, OUT / 'info.json')
grid(cells, ncol=len(cells), size=200, title='saved')

# %% [markdown]
# ## 3. Pack up
#
# Zips `results-lec21/` to `lec21_retune_results.zip`. Download that and bring it back to the laptop.

# %%
import shutil
print(shutil.make_archive('lec21_retune_results', 'zip', root_dir='.', base_dir='results-lec21'))
'''


def build(src, out):
    cells = []
    for chunk in src.strip().split('# %%')[1:]:
        md = chunk.startswith(' [markdown]')
        body = chunk.split('\n', 1)[1].strip('\n') if '\n' in chunk else ''
        if md:
            body = '\n'.join(line[2:] if line.startswith('# ') else line.lstrip('#') for line in body.splitlines())
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': body.splitlines(True)})
        else:
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
                          'source': body.splitlines(True)})
    nb = {'cells': cells, 'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3',
                                                      'language': 'python'},
                                       'language_info': {'name': 'python'}},
          'nbformat': 4, 'nbformat_minor': 5}
    for i, c in enumerate(cells):
        c['id'] = f'c{i:02d}'
    Path(out).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n')
    print('wrote', out, len(cells), 'cells')


build(SRC, sys.argv[1])

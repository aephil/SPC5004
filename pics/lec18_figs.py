"""Pack the lec18 GPU results into the data and images used by lec18.qmd. CPU only, seconds.

The GPU jobs live in pics/lec18_gpu/ (see its README). Unzip what they send back so that the results are at
pics/lec18/raw/results/, then, from the slides folder:

    python pics/lec18_figs.py pack            # pics/lec18/demos-data.js and pics/lec18/*.jpg
    python pics/lec18_figs.py placeholder     # the same from made-up stand-ins, to lay out the deck first

The 890M probe (all out of memory) is kept as pics/lec18/raw/probe-890m.json.
"""
import base64
import csv
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
OUT = HERE / 'lec18'
RAW = OUT / 'raw'
SIZE = 256                     # demo image size; the slides show them at about this size

# Parameter counts for stable-diffusion-v1-5 (measured for lec17).
PARAMS = {'unet': 859_520_964, 'text_encoder': 123_060_480, 'vae': 83_653_863}


def uri(im, size=SIZE, q=82):
    im = im.convert('RGB')
    if size:
        im = im.resize((size, size), Image.LANCZOS) if im.width == im.height else im
    b = io.BytesIO()
    im.save(b, 'JPEG', quality=q)
    return 'data:image/jpeg;base64,' + base64.b64encode(b.getvalue()).decode()


def placeholder_tree(root):
    """Made-up results with the real layout, so the deck can be built before the GPU run."""
    import random
    sys.path.insert(0, str(HERE / 'lec18_gpu'))
    import common as C
    rnd = random.Random(0)

    def card(text, colour=(205, 205, 200), size=512):
        im = Image.new('RGB', (size, size), colour)
        d = ImageDraw.Draw(im)
        for k, line in enumerate(text.split('\n')):
            d.text((20, 20 + 24 * k), line, fill=(60, 60, 60))
        return im

    (root / 'data' / 'examples').mkdir(parents=True, exist_ok=True)
    (root / 'data' / 'thumbs').mkdir(parents=True, exist_ok=True)
    items = []
    for k in range(60):
        f = f'ph{k:03d}.jpg'
        card(f'placeholder print {k}', (200 + k % 40, 190, 170)).save(root / 'data' / 'examples' / f)
        card(f'print {k}', size=160).save(root / 'data' / 'thumbs' / f)
        items.append({'file_name': f, 'caption': f'a placeholder caption for print {k}', 'split': 'train',
                      'title': f'Placeholder title {k}', 'artist': 'Placeholder artist', 'date': 'c. 1800',
                      'accession': f'0000.{k}', 'url': '', 'example': k < 16})
    json.dump({'source': 'PLACEHOLDER', 'n_train': 448, 'n_heldout': 40, 'items': items},
              open(root / 'data' / 'catalogue.json', 'w'))
    stages = [0] + C.STAGES
    keys = [k for k, _ in C.PROMPTS] + [f'train{k}' for k in range(C.N_TRAIN_PROBES)]
    for s in stages:
        for v in (['base'] if s == 0 else ['ema', 'raw']):
            d = root / 'samples' / f's{s:05d}' / v
            d.mkdir(parents=True, exist_ok=True)
            for key in keys:
                for seed in C.SEEDS:
                    shade = int(230 - 120 * (stages.index(s) / (len(stages) - 1)))
                    card(f'PLACEHOLDER\n{key} seed {seed}\nstep {s} {v}', (shade, shade, 230)).save(d / f'{key}-{seed}.jpg')
        ev = {v: {part: {str(t): {'mean': 0.1 + 0.02 * rnd.random() - (0.01 * stages.index(s) / 8 if part == 'train' else 0)}
                         for t in C.EVAL_TIMESTEPS} for part in ('train', 'heldout')}
              for v in (['base'] if s == 0 else ['ema', 'raw'])}
        (root / 'eval').mkdir(exist_ok=True)
        json.dump(ev, open(root / 'eval' / f's{s:05d}.json', 'w'))
    with open(root / 'train_loss.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'wall_time', 'train_loss'])
        for s in range(1, C.STAGES[-1] + 1):
            w.writerow([s, s, abs(rnd.gauss(0.1, 0.06))])
    near = {}
    for p in (root / 'samples').glob('s*/*/*.jpg'):
        near[str(p.relative_to(root))] = [{'file_name': 'ph000.jpg', 'cos': 0.6 + 0.3 * rnd.random()}]
    near['_train_to_train_max_cos'] = {'median': 0.85, 'p95': 0.93}
    json.dump(near, open(root / 'nearest.json', 'w'))
    probe = {'gpu': {'name': 'PLACEHOLDER GPU', 'vram_gb': 80}, 'bnb_8bit_adam_works': True, 'configs': {}}
    import probe_memory as P
    for name, (mixed, ckpt, eight, ema, bs) in P.CONFIGS.items():
        act = (6 if not mixed else 3.5) * (0.35 if ckpt else 1) * bs
        frozen = 0.41 if mixed else 0.83
        opt = 1.72 if eight else 6.88
        e = 3.44 if ema == 'gpu' else 0
        probe['configs'][name] = {'name': name, 'fits': True, 'mixed': mixed, 'ckpt': ckpt, 'eight_bit_adam': eight,
                                  'ema': ema, 'batch': bs, 'mem_frozen_gb': frozen / 1.0737, 'mem_weights_gb': 3.2,
                                  'peak_gb': (frozen + 3.44 * 2 + opt + e + act) / 1.0737,
                                  'sec_per_micro_batch': 0.2 * bs * (1.3 if ckpt else 1) * (1 if mixed else 2.5)}
    if (RAW / 'probe-a100-partial.json').exists():      # real measurements arrived early: use them
        probe = json.load(open(RAW / 'probe-a100-partial.json'))
    json.dump(probe, open(root / 'probe.json', 'w'))
    json.dump({'gpu': {'name': 'PLACEHOLDER GPU', 'vram_gb': 80}, 'n_train': 448,
               'recipe': {'train_batch_size': 4, 'gradient_accumulation_steps': 1, 'gradient_checkpointing': False,
                          'use_8bit_adam': False, 'use_ema': True, 'mixed_precision': 'fp16', 'learning_rate': 1e-5},
               'stages': C.STAGES, 'timing': {str(s): {'train_s': 60.0 * s / 50} for s in C.STAGES}},
              open(root / 'run_info.json', 'w'))


def pack(root, placeholder=False):
    sys.path.insert(0, str(HERE / 'lec18_gpu'))
    import common as C
    run = json.load(open(root / 'run_info.json'))
    cat = json.load(open(root / 'data' / 'catalogue.json'))
    items = {c['file_name']: c for c in cat['items']}
    stages = [0] + [s for s in run['stages'] if (root / 'samples' / f's{s:05d}').exists()]
    train_meta = sorted((c for c in cat['items'] if c['split'] == 'train'), key=lambda c: c['file_name'])
    prompts = [{'key': k, 'text': t} for k, t in C.PROMPTS] + \
              [{'key': f'train{k}', 'text': c['caption'], 'file': c['file_name']}
               for k, c in enumerate(train_meta[:C.N_TRAIN_PROBES])]

    img = {}
    for s in stages:
        for v in (['base'] if s == 0 else ['ema', 'raw']):
            for p in prompts:
                for seed in C.SEEDS:
                    f = root / 'samples' / f's{s:05d}' / v / f'{p["key"]}-{seed}.jpg'
                    if f.exists():
                        img[f'{s}-{"ema" if v == "base" else v}-{p["key"]}-{seed}'] = uri(Image.open(f))
                        if v == 'base':
                            img[f'{s}-raw-{p["key"]}-{seed}'] = img[f'{s}-ema-{p["key"]}-{seed}']

    near = json.load(open(root / 'nearest.json'))
    nearest, thumbs = {}, {}
    for s in stages:
        v = 'base' if s == 0 else 'ema'
        for p in prompts:
            for seed in C.SEEDS:
                r = near.get(f'samples/s{s:05d}/{v}/{p["key"]}-{seed}.jpg')
                if r:
                    nearest[f'{s}-{p["key"]}-{seed}'] = r[0]
                    thumbs.setdefault(r[0]['file_name'], None)
    for p in prompts:
        if 'file' in p:
            thumbs[p['file']] = None
    for f in thumbs:
        thumbs[f] = uri(Image.open(root / 'data' / 'thumbs' / f), size=None)

    examples = [c for c in cat['items'] if c.get('example')][:8]
    data = {'n_train': cat['n_train'], 'n_heldout': cat['n_heldout'], 'source': cat['source'],
            'examples': [{'img': uri(Image.open(root / 'data' / 'examples' / c['file_name']), size=None, q=80),
                          'caption': c['caption'], 'title': c['title'], 'artist': c['artist'],
                          'date': c['date'], 'accession': c['accession']} for c in examples]}

    with open(root / 'train_loss.csv') as f:
        rows = [(int(r['step']), float(r['train_loss'])) for r in csv.DictReader(f)]
    ev = {s: json.load(open(root / 'eval' / f's{s:05d}.json')) for s in stages
          if (root / 'eval' / f's{s:05d}.json').exists()}
    evals = {}
    for s, e in ev.items():
        v = 'base' if s == 0 else 'ema'
        evals[s] = {part: {t: e[v][part][t]['mean'] for t in e[v][part]} for part in ('train', 'heldout')}

    probe = json.load(open(root / 'probe.json'))
    probe_890m = json.load(open(RAW / 'probe-890m.json')) if (RAW / 'probe-890m.json').exists() else None

    steps_per_epoch = run['n_train'] / C.EFFECTIVE_BATCH
    out = {'placeholder': placeholder, 'prompts': prompts, 'seeds': C.SEEDS, 'stages': stages,
           'steps_per_epoch': steps_per_epoch, 'img': img, 'nearest': nearest, 'thumbs': thumbs,
           'train_to_train': near.get('_train_to_train_max_cos'), 'data': data,
           'loss': rows, 'eval': evals, 'eval_t': C.EVAL_TIMESTEPS,
           'params': PARAMS, 'probe': probe, 'probe_890m': probe_890m,
           'run': {'gpu': run['gpu'], 'recipe': run['recipe'], 'timing': run.get('timing', {})}}
    js = '// Generated by pics/lec18_figs.py; do not edit.\nwindow.L18DATA = ' + json.dumps(out, separators=(',', ':')) + ';\n'
    (OUT / 'demos-data.js').write_text(js)
    print('wrote demos-data.js', round(len(js) / 1e6, 2), 'MB;', len(img), 'images')

    # Static images for the slides.
    last = stages[-1]
    for name, s in (('hook-base', 0), ('hook-tuned', last)):
        v = 'base' if s == 0 else 'ema'
        Image.open(root / 'samples' / f's{s:05d}' / v / f'lighthouse-{C.SEEDS[0]}.jpg').convert('RGB').save(
            OUT / f'{name}.jpg', quality=88)
    ex = data['examples'][0]
    Image.open(root / 'data' / 'examples' / examples[0]['file_name']).convert('RGB').save(
        OUT / 'train-example.jpg', quality=85)
    print('wrote hook-base.jpg, hook-tuned.jpg, train-example.jpg  (example:', ex['title'], ')')


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else 'pack'
    OUT.mkdir(exist_ok=True)
    if what == 'placeholder':
        root = RAW / 'placeholder'
        placeholder_tree(root)
        pack(root, placeholder=True)
    else:
        pack(RAW / 'results')


if __name__ == '__main__':
    main()

"""Make the data for the four live demos in lec16.qmd: pics/lec16/demos-data.js.

The demos themselves are in pics/lec16/demos.js (hand-written). As for lecture 15 (pics/lec15_demos.py),
everything is packed into one .js file as data: URIs and numbers, so the deck also works from disk.

Needs numpy, Pillow, torch and diffusers (the base conda env has them; genaibook's scipy no longer
loads on this macOS). The photo is scikit-image's `chelsea` (CC0), read straight from any installed
copy of scikit-image. Run from the slides folder:

    python pics/lec16_demos.py CKPT_DIR      # CKPT_DIR from pics/lec16_train.py, e.g. ~/data/cond-mnist
    python pics/lec16_demos.py               # without the model: the explorer demo is left empty

  sizes     the photo at 256 x 256 and 32 x 32 (schedule x resolution demo)
  classes   three labelled 2D point sets (conditional toy)
  explorer  DDPM samples from the class-conditional MNIST model: 13 conditions (digits 0-9 and three
            blends of the rows for 1 and 7) x 4 starting noises, with x_t and the model's guess at 31
            frames each, as two PNG sprites
"""
import base64
import glob
import importlib.util
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / 'lec16' / 'demos-data.js'

# Frames for the explorer: the timesteps at which x_t and the guess are kept, denser near the end,
# where the detail appears. A final frame holds the finished image.
FRAMES = sorted({int(round(999 * u ** 2)) for u in np.linspace(1, 0, 30)}, reverse=True)
BLENDS = [0.25, 0.5, 0.75]           # of the way from the class-1 row to the class-7 row
SEEDS = 4


def data_uri(im):
    buf = io.BytesIO()
    im.save(buf, 'PNG', optimize=True)
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def chelsea_file():
    spec = importlib.util.find_spec('skimage')
    if spec:
        return Path(spec.origin).parent / 'data' / 'chelsea.png'
    root = Path(sys.base_prefix)
    hits = glob.glob(str(root / 'envs/*/lib/python*/site-packages/skimage/data/chelsea.png'))
    if not hits:
        sys.exit('scikit-image (for its chelsea.png) not found in any conda env')
    return Path(hits[0])


def photos():
    im = Image.open(chelsea_file()).convert('RGB')               # 300 x 451, crop as lec15_16_figs.py
    w, h = im.size
    im = im.crop(((w - h) // 2 + 30, 0, (w - h) // 2 + 30 + h, h))
    big = im.resize((256, 256), Image.LANCZOS)
    small = big.resize((32, 32), Image.BOX)                        # area average, as F.interpolate 'area'
    return data_uri(big), data_uri(small)


def check_schedules():
    """The JS rebuilds both schedules and the timestep embedding; check them against diffusers."""
    import math

    import torch
    from diffusers import DDPMScheduler
    from diffusers.models.embeddings import get_timestep_embedding
    t = np.arange(1000)
    lin = np.cumprod(1 - np.linspace(1e-3, 0.02, 1000))          # the lab's schedule
    f = lambda s: np.cos((s / 1000 + 0.008) / 1.008 * np.pi / 2) ** 2  # noqa: E731
    cos = np.cumprod(np.minimum(1 - f(t + 1) / f(t), 0.999) * -1 + 1)
    ref_lin = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-3, beta_end=0.02).alphas_cumprod.numpy()
    ref_cos = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2').alphas_cumprod.numpy()
    assert np.allclose(lin, ref_lin, atol=1e-6) and np.allclose(cos, ref_cos, atol=1e-6), 'schedule mismatch'
    assert int(np.argmax(np.sqrt(lin) < 0.1)) == 644 and int(np.argmax(np.sqrt(cos) < 0.1)) == 935
    ts = torch.tensor([0, 1, 37, 500, 999])
    ref = get_timestep_embedding(ts, 32, flip_sin_to_cos=True, downscale_freq_shift=0).numpy()
    om = np.exp(-math.log(10000) * np.arange(16) / 16)
    mine = np.concatenate([np.cos(ts.numpy()[:, None] * om), np.sin(ts.numpy()[:, None] * om)], 1)
    assert np.allclose(mine, ref, atol=1e-4), 'embedding mismatch'


def classes(n=300, seed=0):
    """Three labelled 2D shapes: a ring, a line and two blobs; scaled together so each coordinate of
    the whole set has standard deviation 1."""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0, 2 * np.pi, n)
    ring = np.stack([-0.9 + 0.75 * np.cos(a), 0.9 + 0.75 * np.sin(a)], 1)
    u = rng.uniform(0, 1, n)
    line = np.stack([0.2 + 1.5 * u, -1.6 + 1.6 * u], 1)
    k = rng.integers(0, 2, n)
    blobs = np.where(k[:, None] == 0, [-1.2, -1.2], [1.3, 1.4]) + rng.normal(0, 0.2, (n, 2))
    pts = np.concatenate([ring + rng.normal(0, 0.06, (n, 2)), line + rng.normal(0, 0.06, (n, 2)), blobs])
    pts = (pts - pts.mean(0)) / pts.std(0)
    return {'xy': [round(float(v), 3) for v in pts.ravel()], 'label': [0] * n + [1] * n + [2] * n}


def explorer(ckpt):
    """DDPM sampling, written out (it is diffusers' DDPMScheduler.step with its default fixed_small
    variance and clip_sample) so that every condition gets the same fresh noise for a given seed."""
    import torch
    from diffusers import DDPMScheduler, UNet2DModel
    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    unet = UNet2DModel.from_pretrained(ckpt).to(dev).eval()
    W = unet.class_embedding.weight.data
    rows = [W[c] for c in range(10)] + [(1 - a) * W[1] + a * W[7] for a in BLENDS]
    table = torch.nn.Embedding(len(rows), W.shape[1]).to(dev)
    table.weight.data = torch.stack(rows)
    unet.class_embedding = table                                   # labels 10-12 are the blends
    C = len(rows)
    s = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    ab = s.alphas_cumprod.to(dev)
    g = torch.Generator().manual_seed(0)
    start = torch.randn(SEEDS, 1, 32, 32, generator=g)
    x = start.repeat_interleave(C, 0).to(dev)                     # row = seed * C + condition
    labels = torch.arange(C).repeat(SEEDS).to(dev)
    xs, hats = [], []
    with torch.no_grad():
        for t in range(999, -1, -1):
            eps = unet(x, t, class_labels=labels).sample
            a_t, a_p = ab[t], ab[t - 1] if t > 0 else torch.tensor(1.0, device=dev)
            beta = 1 - a_t / a_p
            hat = ((x - (1 - a_t).sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            if t in FRAMES:
                xs.append(x.clamp(-1, 1).cpu()); hats.append(hat.cpu())
            mean = a_p.sqrt() * beta / (1 - a_t) * hat + (1 - beta).sqrt() * (1 - a_p) / (1 - a_t) * x
            if t > 0:
                z = torch.randn(SEEDS, 1, 32, 32, generator=g).repeat_interleave(C, 0).to(dev)
                mean = mean + (beta * (1 - a_p) / (1 - a_t)).sqrt() * z
            x = mean
    xs.append(x.clamp(-1, 1).cpu()); hats.append(x.clamp(-1, 1).cpu())  # the finished image
    # check: one step of the hand-written sampler equals diffusers' own, without noise
    s.set_timesteps(1000)
    with torch.no_grad():
        x1 = start[:1].to(dev); e1 = unet(x1, 500, class_labels=labels[:1]).sample
        ref = s.step(e1.cpu(), 500, x1.cpu()).pred_original_sample
        mine = ((x1 - (1 - ab[500]).sqrt() * e1) / ab[500].sqrt()).clamp(-1, 1).cpu()
    assert torch.allclose(ref, mine, atol=1e-4), 'x0 guess differs from diffusers'

    def sprite(frames):                      # rows: seed * C + condition; columns: frames
        arr = torch.stack(frames, 1)[:, :, 0]                      # (rows, F, 32, 32)
        arr = ((arr + 1) * 127.5).round().byte().numpy()
        R, F = arr.shape[:2]
        return Image.fromarray(arr.transpose(0, 2, 1, 3).reshape(R * 32, F * 32), 'L')
    return {'frames': FRAMES, 'conds': C, 'seeds': SEEDS, 'blends': BLENDS,
            'xt': data_uri(sprite(xs)), 'hat': data_uri(sprite(hats))}


def main():
    check_schedules()
    big, small = photos()
    d = {'photo256': big, 'photo32': small, 'classes': classes()}
    if len(sys.argv) > 1:
        d['explorer'] = explorer(Path(sys.argv[1]).expanduser())
    OUT.write_text('// Generated by pics/lec16_demos.py; do not edit.\nwindow.L16DATA = '
                   + json.dumps(d, separators=(',', ':')) + ';\n')
    print('wrote', OUT, f'{OUT.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()

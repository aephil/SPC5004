"""Make the images, measured numbers and demo data used in lec17.qmd, from Stable Diffusion v1.5.

Needs the `genaibook` env (torch, diffusers, transformers, scikit-image, Pillow, matplotlib) and the
cached `stable-diffusion-v1-5/stable-diffusion-v1-5` fp16 weights (about 2.6 GB, as in lab 17). Runs
offline. Run from the slides folder:

    python pics/lec17_figs.py numbers         # parameter counts, shapes, scheduler -> stdout
    python pics/lec17_figs.py vae             # pics/lec17/vae-*.{jpg,png}; round trip of `astronaut`
    python pics/lec17_figs.py trajectory      # one generation, step by step, with cross-attention maps
    python pics/lec17_figs.py cfg             # one prompt at several guidance scales, two seeds
    python pics/lec17_figs.py steps           # one prompt with 4 to 50 sampling steps
    python pics/lec17_figs.py pack            # pics/lec17/demos-data.js, from the outputs above

Prompts are deliberately not the lab's. The photo is scikit-image's `astronaut` (NASA, public domain),
512 x 512, so the VAE sees it at its native size. Intermediate arrays are kept in pics/lec17/raw/.
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT = HERE / 'lec17'
RAW = OUT / 'raw'
OUT.mkdir(exist_ok=True)
RAW.mkdir(exist_ok=True)
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
MODEL = 'stable-diffusion-v1-5/stable-diffusion-v1-5'

PROMPT_TRAJ = 'a lighthouse on a rocky cliff above the sea at sunset, oil painting'
PROMPT_CFG = 'a red bicycle leaning against a yellow wall'
PROMPT_STEPS = 'a bowl of ramen with chopsticks, photograph'
SCALES = [1, 1.5, 2, 3, 5, 7.5, 10]
SEEDS_CFG = [1, 2]
STEPS = 30


def pipeline():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, variant='fp16',
                                                   safety_checker=None, requires_safety_checker=False,
                                                   local_files_only=True)
    pipe.set_progress_bar_config(disable=True)
    # No enable_attention_slicing(): on MPS its baddbmm gives NaN in fp16.
    return pipe.to(DEV)


def count(m):
    return sum(p.numel() for p in m.parameters())


def to_img(x):
    """1x3xHxW in [-1, 1] -> PIL."""
    x = ((x[0].float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(x)


def strip(images, size, path, gap=6):
    out = Image.new('RGB', (len(images) * size + (len(images) - 1) * gap, size), 'white')
    for k, im in enumerate(images):
        out.paste(im.resize((size, size), Image.LANCZOS), (k * (size + gap), 0))
    out.save(path, quality=88)
    print('wrote', path, out.size)


def latent_rgb(z):
    """A 4x64x64 latent as a false-colour 64x64 image: each channel rescaled, channels 0-2 as R, G, B."""
    z = z[0].float().cpu().numpy()[:3]
    lo, hi = np.percentile(z, 1, axis=(1, 2), keepdims=True), np.percentile(z, 99, axis=(1, 2), keepdims=True)
    z = np.clip((z - lo) / (hi - lo), 0, 1)
    return Image.fromarray((z.transpose(1, 2, 0) * 255).astype(np.uint8))


def channel_imgs(z, size=192):
    """Each of the 4 latent channels as a grey image, scaled to its own 1st-99th percentile."""
    import matplotlib
    cmap = matplotlib.colormaps['viridis']
    out = []
    for c in z[0].float().cpu().numpy():
        lo, hi = np.percentile(c, [1, 99])
        rgb = (cmap(np.clip((c - lo) / (hi - lo), 0, 1))[:, :, :3] * 255).astype(np.uint8)
        out.append(Image.fromarray(rgb).resize((size, size), Image.NEAREST))
    return out


@torch.inference_mode()
def encode_text(pipe, prompts):
    ids = pipe.tokenizer(prompts, padding='max_length', max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors='pt').input_ids.to(DEV)
    return pipe.text_encoder(ids)[0]


@torch.inference_mode()
def decode(pipe, z):
    return pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample


def numbers():
    pipe = pipeline()
    print('scheduler:', type(pipe.scheduler).__name__, dict(pipe.scheduler.config))
    for name in ['text_encoder', 'unet', 'vae']:
        print(f'{name}: {count(getattr(pipe, name)):,}')
    print(f'vae encoder: {count(pipe.vae.encoder) + count(pipe.vae.quant_conv):,}  '
          f'decoder: {count(pipe.vae.decoder) + count(pipe.vae.post_quant_conv):,}')
    print('scaling_factor', pipe.vae.config.scaling_factor)
    print('tokenizer max length', pipe.tokenizer.model_max_length)
    toks = pipe.tokenizer(PROMPT_TRAJ).input_ids
    print('prompt tokens', len(toks), [pipe.tokenizer.decoder[t] for t in toks])
    print('text embeddings', tuple(encode_text(pipe, [PROMPT_TRAJ]).shape))
    attn2 = [(n, m) for n, m in pipe.unet.named_modules() if n.endswith('attn2')]
    print('cross-attention layers', len(attn2))
    a = attn2[0][1]
    print('first attn2: heads', a.heads, 'to_q', tuple(a.to_q.weight.shape), 'to_k', tuple(a.to_k.weight.shape))
    print(f'unet cross-attention params: {sum(count(m) for _, m in attn2):,}')
    pipe.scheduler.set_timesteps(STEPS)
    print('timesteps', pipe.scheduler.timesteps.tolist())
    print('init_noise_sigma', pipe.scheduler.init_noise_sigma)


def astronaut():
    import importlib.util
    path = Path(importlib.util.find_spec('skimage').origin).parent / 'data' / 'astronaut.png'
    return Image.open(path).convert('RGB').resize((512, 512))


@torch.inference_mode()
def vae():
    pipe = pipeline()
    im = astronaut()
    x = torch.tensor(np.asarray(im), dtype=torch.float16, device=DEV).permute(2, 0, 1)[None] / 127.5 - 1
    z = pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor
    print('latent', tuple(z.shape), 'std', float(z.float().std()))
    rec = to_img(decode(pipe, z))
    im.save(OUT / 'vae-in.jpg', quality=92)
    rec.save(OUT / 'vae-out.jpg', quality=92)
    for k, c in enumerate(channel_imgs(z)):
        c.save(RAW / f'vae-ch{k}.png')
    strip(channel_imgs(z), 192, OUT / 'vae-channels.png')
    err = np.abs(np.asarray(im, float) - np.asarray(rec, float)).mean()
    print(f'mean abs pixel error {err:.2f} / 255')
    # Pure N(0, I) latents through the decoder: not a generator on its own.
    g = torch.Generator('cpu').manual_seed(0)
    rand = [to_img(decode(pipe, torch.randn(1, 4, 64, 64, generator=g).to(DEV, torch.float16)))
            for _ in range(3)]
    strip(rand, 256, OUT / 'vae-random.jpg')


class ChunkedAttn:
    """Plain attention, computed for 1,024 queries at a time with a float32 softmax: much lower peak
    memory than one 4,096 x 4,096 matrix, and no NaN (unlike enable_attention_slicing on MPS)."""
    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 **kw):
        ctx = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(ctx))
        v = attn.head_to_batch_dim(attn.to_v(ctx))
        out = torch.empty_like(q)
        for i in range(0, q.shape[1], 1024):
            sc = torch.bmm(q[:, i:i + 1024].float(), k.float().transpose(1, 2)) * attn.scale
            out[:, i:i + 1024] = torch.bmm(sc.softmax(-1).to(v.dtype), v)
        return attn.to_out[1](attn.to_out[0](attn.batch_to_head_dim(out)))


def low_memory(pipe):
    pipe.unet.set_attn_processor(ChunkedAttn())
    return pipe


class CrossAttnStore:
    """An attention processor for attn2 layers that computes the softmax explicitly, so that the
    attention weights from image positions to prompt tokens can be recorded (16 x 16 layers only)."""
    def __init__(self, sink):
        self.sink = sink

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None,
                 **kw):
        b, n, _ = hidden_states.shape
        q = attn.head_to_batch_dim(attn.to_q(hidden_states))
        k = attn.head_to_batch_dim(attn.to_k(encoder_hidden_states))
        v = attn.head_to_batch_dim(attn.to_v(encoder_hidden_states))
        # Not attn.get_attention_scores: its baddbmm over an uninitialised tensor gives NaN on MPS.
        scores = torch.bmm(q.float(), k.float().transpose(1, 2)) * attn.scale
        probs = scores.softmax(-1).to(q.dtype)                        # (b * heads, n, 77)
        if self.sink.get('on') and n == 256:
            p = probs.view(b, attn.heads, n, -1)[b // 2:].float().mean(1)  # conditional half, mean of heads
            self.sink['maps'].append(p[0].cpu())
        out = attn.batch_to_head_dim(torch.bmm(probs, v))
        return attn.to_out[1](attn.to_out[0](out))


@torch.inference_mode()
def sample(pipe, prompt, scale, seed, steps=STEPS, record=None, sink=None):
    """The lab's hand-written loop (both passes batched), with optional per-step records."""
    cond = encode_text(pipe, [prompt])
    uncond = encode_text(pipe, [''])
    emb = torch.cat([uncond, cond])
    pipe.scheduler.set_timesteps(steps)
    g = torch.Generator('cpu').manual_seed(seed)
    z = torch.randn(1, 4, 64, 64, generator=g).to(DEV, torch.float16) * pipe.scheduler.init_noise_sigma
    ab = pipe.scheduler.alphas_cumprod.to(DEV)
    for i, t in enumerate(pipe.scheduler.timesteps):
        if sink is not None:
            sink['on'] = True
        inp = pipe.scheduler.scale_model_input(torch.cat([z] * 2), t)
        e_u, e_c = pipe.unet(inp, t, encoder_hidden_states=emb).sample.chunk(2)
        if sink is not None:
            sink['on'] = False
        e = e_u + scale * (e_c - e_u)
        if record is not None:
            a = ab[t].to(torch.float16)
            x0 = (z - (1 - a).sqrt() * e) / a.sqrt()
            record.append((int(t), z.clone(), x0))
        z = pipe.scheduler.step(e, t, z).prev_sample
    return z


def trajectory():
    pipe = pipeline()
    sink = {'maps': [], 'on': False}
    store = CrossAttnStore(sink)
    pipe.unet.set_attn_processor({n: (store if '.attn2.' in n else p)
                                  for n, p in pipe.unet.attn_processors.items()})
    rec = []
    z = sample(pipe, PROMPT_TRAJ, 7.5, seed=3, record=rec, sink=sink)
    final = to_img(decode(pipe, z))
    final.save(OUT / 'traj-final.jpg', quality=92)
    keep = list(range(0, STEPS, 2)) + [STEPS - 1]                  # 16 frames
    frames = []
    for i in keep:
        t, zt, x0 = rec[i]
        frames.append({'t': t, 'i': i,
                       'xt': to_img(decode(pipe, zt)), 'x0': to_img(decode(pipe, x0)), 'z': latent_rgb(zt)})
    frames.append({'t': 0, 'i': STEPS, 'xt': final, 'x0': final, 'z': latent_rgb(z)})
    for f in frames:
        for k in ['xt', 'x0']:
            f[k].resize((256, 256), Image.LANCZOS).save(RAW / f"traj-{k}-{f['i']:02d}.jpg", quality=85)
        f['z'].save(RAW / f"traj-z-{f['i']:02d}.png")
    json.dump([{'t': f['t'], 'i': f['i']} for f in frames], open(RAW / 'traj.json', 'w'))
    strip([frames[k]['xt'] for k in [0, 4, 8, 12, 16]], 256, OUT / 'traj-xt.jpg')
    strip([frames[k]['x0'] for k in [0, 4, 8, 12, 16]], 256, OUT / 'traj-x0.jpg')
    # Cross-attention: mean over every 16 x 16 layer and every step.
    maps = torch.stack(sink['maps']).mean(0)                         # (256, 77)
    ids = pipe.tokenizer(PROMPT_TRAJ).input_ids
    words = [pipe.tokenizer.decoder[t].replace('</w>', '') for t in ids]
    print(len(sink['maps']), 'maps recorded; tokens', words)
    out = {'words': words, 'maps': []}
    for j in range(len(ids)):
        m = maps[:, j].view(16, 16).numpy()
        out['maps'].append([round(float(v), 5) for v in m.ravel()])
    json.dump(out, open(RAW / 'xattn.json', 'w'))
    print('share of attention on <startoftext>:', float(maps[:, 0].mean()))


def cfg():
    pipe = low_memory(pipeline())
    for seed in SEEDS_CFG:
        for s in SCALES:
            path = RAW / f'cfg-{seed}-{s}.jpg'
            if path.exists():                                        # resume an interrupted run
                continue
            im = to_img(decode(pipe, sample(pipe, PROMPT_CFG, s, seed)))
            im.resize((320, 320), Image.LANCZOS).save(path, quality=86)
            print('cfg', seed, s, flush=True)


def steps():
    pipe = low_memory(pipeline())
    ims = []
    for n in [4, 8, 15, 30, 50]:
        ims.append(to_img(decode(pipe, sample(pipe, PROMPT_STEPS, 7.5, seed=4, steps=n))))
        print('steps', n, flush=True)
    strip(ims, 256, OUT / 'steps.jpg')


def uri(path):
    kind = 'png' if path.suffix == '.png' else 'jpeg'
    return f'data:image/{kind};base64,' + base64.b64encode(path.read_bytes()).decode()


def pack():
    d = {}
    if (OUT / 'vae-in.jpg').exists():
        d['vae'] = {'in': uri(OUT / 'vae-in.jpg'), 'out': uri(OUT / 'vae-out.jpg'),
                    'ch': [uri(RAW / f'vae-ch{k}.png') for k in range(4)]}
    if (RAW / 'traj.json').exists():
        fr = json.load(open(RAW / 'traj.json'))
        d['traj'] = [{'t': f['t'], 'i': f['i'],
                      'xt': uri(RAW / f"traj-xt-{f['i']:02d}.jpg"), 'x0': uri(RAW / f"traj-x0-{f['i']:02d}.jpg"),
                      'z': uri(RAW / f"traj-z-{f['i']:02d}.png")} for f in fr]
        d['traj_steps'] = STEPS
        d['xattn'] = json.load(open(RAW / 'xattn.json'))
        d['xattn']['image'] = uri(OUT / 'traj-final.jpg')
    # Only the scales done for the first seed, and only seeds done at all of those scales.
    scales = [s for s in SCALES if (RAW / f'cfg-{SEEDS_CFG[0]}-{s}.jpg').exists()]
    seeds = [seed for seed in SEEDS_CFG if all((RAW / f'cfg-{seed}-{s}.jpg').exists() for s in scales)]
    if scales:
        d['cfg'] = {'scales': scales, 'seeds': seeds, 'prompt': PROMPT_CFG,
                    'img': {f'{seed}-{s}': uri(RAW / f'cfg-{seed}-{s}.jpg') for seed in seeds for s in scales}}
        print('cfg scales', scales, 'seeds', seeds)
    sys.path.insert(0, str(HERE))
    from lec16_demos import classes                                    # the same toy data as lec16
    d['classes'] = classes()
    path = OUT / 'demos-data.js'
    path.write_text('// Generated by pics/lec17_figs.py pack; do not edit.\nwindow.L17DATA = '
                    + json.dumps(d, separators=(',', ':')) + ';\n')
    print('wrote', path, f'{path.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    {'numbers': numbers, 'vae': vae, 'trajectory': trajectory, 'cfg': cfg, 'steps': steps,
     'pack': pack}[sys.argv[1]]()

"""Generate the fixed prompts and measure a fixed-noise loss for one training stage.

    python sample_stage.py STEP

STEP 0 is the base Stable Diffusion v1.5 UNet. Otherwise the UNet comes from work/run/checkpoint-STEP, in two
variants: `ema` (the averaged weights, what the finished model uses) and `raw` (the weights being trained).

Images: results/samples/sSTEP/{base|ema|raw}/KEY-SEED.jpg, the same starting noise for each seed at every stage.
Loss:   results/eval/sSTEP.json, the training loss (MSE between predicted and actual noise) on EVAL_N_TRAIN
        training prints and all held-out prints, at fixed timesteps with fixed noise, so stages are comparable.
Resumable: existing images are skipped.
"""
import json
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import (BASE, DATA, EVAL_N_TRAIN, EVAL_TIMESTEPS, GUIDANCE, RESULTS, RUN, SAMPLE_STEPS, SEEDS,
                    probes, save_json)

DEV = torch.device('cuda')
DT = torch.float16


def load_pipe():
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=DT, safety_checker=None,
                                                   requires_safety_checker=False).to(DEV)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def set_unet(pipe, path):
    from diffusers import UNet2DConditionModel
    unet = UNet2DConditionModel.from_pretrained(path, torch_dtype=DT).to(DEV)
    pipe.unet = unet
    return pipe


def generate(pipe, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for key, text in probes():
        for seed in SEEDS:
            path = out_dir / f'{key}-{seed}.jpg'
            if path.exists():
                continue
            lat = torch.randn((1, 4, 64, 64), generator=torch.Generator('cpu').manual_seed(seed))
            img = pipe(text, latents=lat.to(DEV, DT), num_inference_steps=SAMPLE_STEPS,
                       guidance_scale=GUIDANCE).images[0]
            img.save(path, quality=90)
            n += 1
    return n


def eval_set(pipe):
    """Latents, text embeddings and fixed noise for the evaluation prints; cached on first use."""
    cache = DATA / 'eval_set.pt'
    if cache.exists():
        return torch.load(cache)
    sets = {}
    for part, limit in (('train', EVAL_N_TRAIN), ('heldout', None)):
        meta = sorted((json.loads(l) for l in (DATA / part / 'metadata.jsonl').read_text().splitlines()),
                      key=lambda m: m['file_name'])[:limit]
        lats, embs = [], []
        for m in meta:
            im = Image.open(DATA / part / m['file_name']).convert('RGB')
            s = min(im.size)
            im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
            x = torch.from_numpy(np.asarray(im.resize((512, 512), Image.BICUBIC))).float()
            x = (x.permute(2, 0, 1)[None] / 127.5 - 1).to(DEV, DT)
            ids = pipe.tokenizer(m['text'], padding='max_length', max_length=77, truncation=True,
                                 return_tensors='pt').input_ids.to(DEV)
            with torch.no_grad():
                lats.append(pipe.vae.encode(x).latent_dist.mean * pipe.vae.config.scaling_factor)
                embs.append(pipe.text_encoder(ids)[0])
        lat = torch.cat(lats).cpu()
        g = torch.Generator('cpu').manual_seed(1234)
        noise = torch.randn((len(EVAL_TIMESTEPS),) + tuple(lat.shape), generator=g)
        sets[part] = {'lat': lat, 'emb': torch.cat(embs).cpu(), 'noise': noise,
                      'files': [m['file_name'] for m in meta]}
    torch.save(sets, cache)
    return sets


@torch.no_grad()
def eval_loss(pipe, sets):
    from diffusers import DDPMScheduler
    sched = DDPMScheduler.from_pretrained(BASE, subfolder='scheduler')
    res = {}
    for part, s in sets.items():
        res[part] = {}
        for k, t in enumerate(EVAL_TIMESTEPS):
            losses = []
            for i in range(0, len(s['lat']), 8):
                lat = s['lat'][i:i + 8].to(DEV, torch.float32)
                noise = s['noise'][k, i:i + 8].to(DEV, torch.float32)
                tt = torch.full((len(lat),), t, device=DEV, dtype=torch.long)
                noisy = sched.add_noise(lat, noise, tt)
                pred = pipe.unet(noisy.to(DT), tt, s['emb'][i:i + 8].to(DEV, DT)).sample
                losses += F.mse_loss(pred.float(), noise, reduction='none').mean(dim=(1, 2, 3)).tolist()
            res[part][str(t)] = {'mean': sum(losses) / len(losses), 'per_image': losses}
    return res


def main():
    step = int(sys.argv[1])
    t0 = time.time()
    pipe = load_pipe()
    sets = eval_set(pipe)
    variants = {'base': None} if step == 0 else {'ema': RUN / f'checkpoint-{step}' / 'unet_ema',
                                                  'raw': RUN / f'checkpoint-{step}' / 'unet'}
    out = RESULTS / 'samples' / f's{step:05d}'
    ev_path = RESULTS / 'eval' / f's{step:05d}.json'
    ev = json.loads(ev_path.read_text()) if ev_path.exists() else {}
    for name, path in variants.items():
        if path is not None:
            set_unet(pipe, path)
        n = generate(pipe, out / name)
        if name not in ev:
            ev[name] = eval_loss(pipe, sets)
            save_json(ev, ev_path)
        print(f'stage {step} {name}: {n} new images; held-out loss (t=400) '
              f'{ev[name]["heldout"]["400"]["mean"]:.4f}, train {ev[name]["train"]["400"]["mean"]:.4f}', flush=True)
    print(f'stage {step} sampled in {time.time() - t0:.0f} s')


if __name__ == '__main__':
    main()

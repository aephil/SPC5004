"""Measure what a full Stable Diffusion v1.5 UNet fine-tune needs on this GPU: peak memory and time per step
for each combination of the usual memory-saving options, and the time to generate one image.

    python probe_memory.py            # runs every configuration, each in a fresh process; ~15-20 min
    python probe_memory.py --one NAME # (internal) run a single configuration

Writes results/probe.json. A configuration that runs out of memory is recorded as such, not as a failure.
Synthetic data (random images, a fixed caption): memory and speed don't depend on what the pictures show.
Each training step mirrors train_text_to_image.py: VAE encode (frozen), add noise, text encode (frozen),
UNet forward and backward, optimiser step, zero_grad, then the EMA update.
"""
import argparse
import gc
import json
import statistics
import subprocess
import sys
import time

from common import BASE, RESULTS, gpu_info, have_bnb, save_json

# name: (mixed precision, gradient checkpointing, 8-bit Adam, EMA: None / 'gpu' / 'cpu', micro-batch)
CONFIGS = {
    # A full grid of precision x checkpointing x batch, with plain AdamW and no EMA: activation memory for
    # every combination (it doesn't depend on the optimiser or EMA).
    'fp32':                    (False, False, False, None, 1),
    'fp32+ckpt':               (False, True, False, None, 1),
    'fp32/bs4':                (False, False, False, None, 4),
    'fp32+ckpt/bs4':           (False, True, False, None, 4),
    'mixed':                   (True, False, False, None, 1),
    'mixed+ckpt':              (True, True, False, None, 1),
    'mixed/bs4':               (True, False, False, None, 4),
    'mixed+ckpt/bs4':          (True, True, False, None, 4),
    # The savers stacked up, as real recipes use them.
    'fp32+ema':                (False, False, False, 'gpu', 1),
    'mixed+ckpt+ema':          (True, True, False, 'gpu', 1),
    'mixed+ckpt+8bit':         (True, True, True, None, 1),
    'mixed+ckpt+8bit+ema':     (True, True, True, 'gpu', 1),
    'mixed+ckpt+8bit+ema_cpu': (True, True, True, 'cpu', 1),
    'mixed+ckpt+8bit/bs4':     (True, True, True, None, 4),
}
STEPS, WARMUP = 10, 3


def run_one(name):
    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from diffusers.training_utils import EMAModel
    from transformers import CLIPTextModel, CLIPTokenizer

    mixed, ckpt, eight, ema_where, bs = CONFIGS[name]
    dev = torch.device('cuda')
    GB = 2**30
    mem = lambda: torch.cuda.memory_allocated() / GB          # noqa: E731
    out = {'name': name, 'mixed': mixed, 'ckpt': ckpt, 'eight_bit_adam': eight, 'ema': ema_where, 'batch': bs}

    frozen_dtype = torch.float16 if mixed else torch.float32
    tok = CLIPTokenizer.from_pretrained(BASE, subfolder='tokenizer')
    te = CLIPTextModel.from_pretrained(BASE, subfolder='text_encoder').to(dev, frozen_dtype).requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(BASE, subfolder='vae').to(dev, frozen_dtype).requires_grad_(False)
    out['mem_frozen_gb'] = mem()
    unet = UNet2DConditionModel.from_pretrained(BASE, subfolder='unet').to(dev)   # trained weights stay fp32
    unet.train()
    if ckpt:
        unet.enable_gradient_checkpointing()
    out['mem_weights_gb'] = mem() - out['mem_frozen_gb']
    params = [p for p in unet.parameters() if p.requires_grad]
    if eight:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=1e-5)
    else:
        opt = torch.optim.AdamW(params, lr=1e-5)
    ema = None
    if ema_where:
        ema = EMAModel(unet.parameters(), model_cls=UNet2DConditionModel, model_config=unet.config)
        ema.to('cpu' if ema_where == 'cpu' else dev)
    out['mem_ema_gb'] = mem() - out['mem_frozen_gb'] - out['mem_weights_gb']
    sched = DDPMScheduler.from_pretrained(BASE, subfolder='scheduler')
    scaler = torch.amp.GradScaler('cuda', enabled=mixed)
    ids = tok(['a woodblock print of a wave'] * bs, padding='max_length', max_length=77,
              truncation=True, return_tensors='pt').input_ids.to(dev)
    g = torch.Generator(device='cpu').manual_seed(0)

    times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(STEPS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        px = (torch.rand(bs, 3, 512, 512, generator=g) * 2 - 1).to(dev, frozen_dtype)
        with torch.no_grad():
            lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
            enc = te(ids)[0]
        noise = torch.randn_like(lat)
        t = torch.randint(0, 1000, (bs,), device=dev)
        noisy = sched.add_noise(lat, noise, t)
        with torch.autocast('cuda', dtype=torch.float16, enabled=mixed):
            pred = unet(noisy.float() if not mixed else noisy, t, enc).sample
        loss = F.mse_loss(pred.float(), noise.float())
        scaler.scale(loss).backward()
        if step == 0:
            out['mem_grads_gb'] = sum(p.grad.numel() * p.grad.element_size() for p in params if p.grad is not None) / GB
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step == 0:
            out['mem_after_step_gb'] = mem()   # frozen + weights + optimiser state (+ EMA if on GPU)
            out['mem_optimiser_gb'] = mem() - out['mem_frozen_gb'] - out['mem_weights_gb'] - out['mem_ema_gb']
        if ema is not None:
            if ema_where == 'cpu':
                ema.to(dev, non_blocking=True)
            ema.step(unet.parameters())
            if ema_where == 'cpu':
                ema.to('cpu', non_blocking=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        if not torch.isfinite(loss):
            out['note'] = 'non-finite loss'
    out['peak_gb'] = torch.cuda.max_memory_allocated() / GB
    out['peak_reserved_gb'] = torch.cuda.max_memory_reserved() / GB
    out['sec_per_micro_batch'] = statistics.median(times[WARMUP:])
    out['loss_last'] = float(loss)
    out['fits'] = True
    print(json.dumps(out))


def time_sampling():
    import torch
    from diffusers import StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(BASE, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False).to('cuda')
    pipe.set_progress_bar_config(disable=True)
    pipe('warm up', num_inference_steps=2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    img = pipe('a woodblock print of a wave', num_inference_steps=30, guidance_scale=7.5,
               generator=torch.Generator('cpu').manual_seed(0)).images[0]
    torch.cuda.synchronize()
    RESULTS.mkdir(parents=True, exist_ok=True)
    img.save(RESULTS / 'probe-sample.jpg', quality=90)
    print(json.dumps({'sec_per_image_30_steps': time.perf_counter() - t0,
                      'peak_gb': torch.cuda.max_memory_allocated() / 2**30}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--one')
    ap.add_argument('--sample', action='store_true')
    a = ap.parse_args()
    if a.one:
        return run_one(a.one)
    if a.sample:
        return time_sampling()

    path = RESULTS / 'probe.json'
    res = json.loads(path.read_text()) if path.exists() else {}
    res['gpu'] = gpu_info()
    res['bnb_8bit_adam_works'] = have_bnb()
    print(res['gpu'], '8-bit Adam works:', res['bnb_8bit_adam_works'], flush=True)
    res.setdefault('configs', {})
    for name, cfg in CONFIGS.items():
        if name in res['configs']:
            print('skip (done):', name)
            continue
        if cfg[2] and not res['bnb_8bit_adam_works']:
            res['configs'][name] = {'name': name, 'fits': None, 'note': '8-bit Adam unavailable'}
            continue
        print('running', name, '...', flush=True)
        p = subprocess.run([sys.executable, __file__, '--one', name], capture_output=True, text=True, timeout=3600)
        last = [l for l in p.stdout.splitlines() if l.startswith('{')]
        if p.returncode == 0 and last:
            r = json.loads(last[-1])
        else:
            err = p.stderr[-3000:]
            oom = 'out of memory' in err.lower() or 'OutOfMemoryError' in err
            r = {'name': name, 'fits': False if oom else None, 'error': err[-600:]}
        res['configs'][name] = r
        print('  ', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != 'error'}, flush=True)
        save_json(res, path)
        gc.collect()
    if 'sampling' not in res:
        print('timing one 30-step generation ...', flush=True)
        p = subprocess.run([sys.executable, __file__, '--sample'], capture_output=True, text=True, timeout=3600)
        last = [l for l in p.stdout.splitlines() if l.startswith('{')]
        res['sampling'] = json.loads(last[-1]) if last else {'error': p.stderr[-600:]}
        print('  ', res['sampling'])
    save_json(res, path)
    print('wrote', path)


if __name__ == '__main__':
    main()

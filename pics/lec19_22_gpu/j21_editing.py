"""lec21: prompt weighting, semantic guidance, inversion, InstructPix2Pix. ~25-40 min on a mid-range GPU.

results/lec21/weight/<key>-w<weight>-<seed>.jpg
results/lec21/sega/<edit>-g<scale>-th<threshold>-<seed>.jpg, plus sega/maps-<name>.npy (where each edit acts, per step)
results/lec21/invert/*.jpg     random noise vs DDIM inversion vs LEDITS++ inversion, reconstructions and edits
results/lec21/pix2pix/<key>/sI<..>-sT<..>.jpg
Each section is independent: one failing doesn't stop the others.
"""
import numpy as np
import torch

from common import (NEGATIVE, PIX2PIX, QUICK, RESULTS, STEPS, Log, Timer, finish, gen, inp, load_json, run,
                    save_img, save_json, sd_pipe)

log = Log('lec21')
SEGA_STEPS, WARMUP, INV_STEPS, P2P_STEPS = (12, 3, 8, 4) if QUICK else (50, 10, 50, 20)
OUT = RESULTS / 'lec21'
info = load_json(OUT / 'info.json', {})


def save_info():
    save_json(info, OUT / 'info.json')


# ---------------------------------------------------------------- prompt weighting (compel)
WEIGHTS = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0]
WPROMPTS = [('snow', 'a cottage in a forest, ({}){}, warm light in the windows, photograph', 'snowy'),
            ('red', 'a vintage car parked on a street, ({}){}, photograph', 'red')]


@run
def weighting():
    from compel import Compel
    pipe = sd_pipe()
    compel = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder)
    neg = compel(NEGATIVE)
    info['weight'] = {'weights': WEIGHTS, 'prompts': {k: tpl.format(wd, 'w') for k, tpl, wd in WPROMPTS}}
    # How compel weights a token: z_w = z_empty + w (z - z_empty), position by position. Record vector norms.
    with torch.no_grad():
        enc = lambda p: pipe.text_encoder(pipe.tokenizer(p, padding='max_length', max_length=77,   # noqa: E731
                                                         return_tensors='pt').input_ids.cuda())[0][0].float()
        p0 = 'a cottage in a forest, snowy, warm light in the windows, photograph'
        z, e = enc(p0), enc('')
        toks = pipe.tokenizer.convert_ids_to_tokens(pipe.tokenizer(p0).input_ids)
        info['weight']['tokens'] = toks
        info['weight']['norm_z'] = [round(float(v), 2) for v in z[:len(toks)].norm(dim=-1)]
        info['weight']['norm_z_minus_empty'] = [round(float(v), 2) for v in (z - e)[:len(toks)].norm(dim=-1)]
    save_info()
    with Timer() as t:
        n = 0
        for key, template, word in WPROMPTS:
            for w in WEIGHTS:
                for seed in [3, 4]:
                    f = OUT / 'weight' / f'{key}-w{w}-{seed}.jpg'
                    if f.exists():
                        continue
                    c = compel(template.format(word, w))
                    c, nn = compel.pad_conditioning_tensors_to_same_length([c, neg])
                    im = pipe(prompt_embeds=c, negative_prompt_embeds=nn, num_inference_steps=STEPS,
                              generator=gen(seed)).images[0]
                    save_img(im, f)
                    n += 1
    if n:
        log.time('weight', t.s, n)


# ---------------------------------------------------------------- semantic guidance (SEGA)
SEGA_PROMPT = 'a photo of a quiet street in a small town, houses and trees'
EDITS = {'autumn': ('autumn, orange leaves', False), 'snow': ('snow, winter', False),
         'cars': ('cars, parked cars', True)}         # True = push away from the concept
SEGA_SEEDS = [21, 22]


@run
def sega():
    from diffusers import SemanticStableDiffusionPipeline
    pipe = sd_pipe(SemanticStableDiffusionPipeline)
    info['sega'] = {'prompt': SEGA_PROMPT, 'edits': EDITS, 'steps': SEGA_STEPS, 'warmup': WARMUP,
                    'momentum': [0.3, 0.6]}
    save_info()

    def gen_one(edits, scale, threshold, seed):
        kw = {}
        if edits:
            kw = dict(editing_prompt=[EDITS[e][0] for e in edits],
                      reverse_editing_direction=[EDITS[e][1] for e in edits],
                      edit_guidance_scale=[scale] * len(edits), edit_threshold=[threshold] * len(edits),
                      edit_warmup_steps=[WARMUP] * len(edits), edit_momentum_scale=0.3, edit_mom_beta=0.6)
        return pipe(SEGA_PROMPT, negative_prompt=NEGATIVE, num_inference_steps=SEGA_STEPS, guidance_scale=7.5,
                    generator=gen(seed), **kw).images[0]

    jobs = [((), 0, 0.95, s) for s in SEGA_SEEDS]                                         # no edit
    jobs += [(('autumn',), g, 0.95, s) for g in [2, 4, 6, 8, 12] for s in SEGA_SEEDS]       # strength sweep
    jobs += [(('autumn',), 6, th, SEGA_SEEDS[0]) for th in [0.0, 0.5, 0.8, 0.9, 0.99]]      # threshold sweep
    jobs += [((e,), 6, 0.95, s) for e in ['snow', 'cars'] for s in SEGA_SEEDS]
    jobs += [(('autumn', 'cars'), 6, 0.95, s) for s in SEGA_SEEDS]                          # two edits at once
    with Timer() as t:
        n = 0
        for edits, g, th, seed in jobs:
            name = ('+'.join(edits) or 'none') + f'-g{g}-th{th}-{seed}'
            f = OUT / 'sega' / f'{name}.jpg'
            if f.exists():
                continue
            save_img(gen_one(edits, g, th, seed), f)
            n += 1
            if edits and pipe.sem_guidance is not None and seed == SEGA_SEEDS[0]:
                # |edit guidance| summed over the 4 latent channels: a 64 x 64 map per step of where the edit acts
                m = pipe.sem_guidance.float().abs().sum(dim=-3).squeeze().numpy().astype(np.float16)
                np.save(OUT / 'sega' / f'maps-{name}.npy', m)
    if n:
        log.time('sega', t.s, n)


# ---------------------------------------------------------------- inversion
CAT = 'a photo of a cat'
INV = OUT / 'invert'


@run
def ddim_inversion():
    from diffusers import DDIMScheduler
    photo = inp('cat.png')
    pipe = sd_pipe()
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    @torch.no_grad()
    def encode_text(p):
        ids = pipe.tokenizer(p, padding='max_length', max_length=77, return_tensors='pt').input_ids.cuda()
        return pipe.text_encoder(ids)[0]

    @torch.no_grad()
    def ddim_invert(z0, cond, steps=INV_STEPS):
        """DDIM run backwards (image -> noise), no guidance: each step assumes the noise prediction at the
        current point also holds at the next, noisier one."""
        s = pipe.scheduler
        s.set_timesteps(steps)
        ts = s.timesteps.flip(0)                                    # small t -> large t
        ab = s.alphas_cumprod.to('cuda')
        z = z0.float()
        for i, t in enumerate(ts):
            eps = pipe.unet(z.half(), t, encoder_hidden_states=cond).sample.float()
            a = ab[ts[i - 1]] if i > 0 else s.final_alpha_cumprod.to('cuda')
            a_next = ab[t]
            x0 = (z - (1 - a).sqrt() * eps) / a.sqrt()
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        return z.half()

    with torch.no_grad():
        z0 = pipe.vae.encode(pipe.image_processor.preprocess(photo).to('cuda', torch.float16)).latent_dist.mean
    z0 = z0 * pipe.vae.config.scaling_factor
    with Timer() as t:
        if not (INV / 'ddim-w7.5.jpg').exists():
            zT = ddim_invert(z0, encode_text(CAT))
            info['inversion'] = {'zT_std': round(float(zT.float().std()), 3),
                                 'zT_mean': round(float(zT.float().mean()), 3), 'steps': INV_STEPS}
            save_info()
            for w in [1.0, 3.0, 7.5]:
                im = pipe(CAT, latents=zT, num_inference_steps=INV_STEPS, guidance_scale=w, eta=0.0).images[0]
                save_img(im, INV / f'ddim-w{w}.jpg')
            # The inverted noise, decoded as if it were an image, beside ordinary random noise
            rnd = torch.randn(z0.shape, generator=gen(0), device='cuda', dtype=z0.dtype)
            for name, z in [('zT', zT), ('random', rnd)]:
                with torch.no_grad():
                    x = pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample
                save_img(pipe.image_processor.postprocess(x, output_type='pil')[0], INV / f'noise-{name}.jpg')
        for seed in [1, 2]:                                                 # random noise, same prompt
            f = INV / f'random-{seed}.jpg'
            if not f.exists():
                save_img(pipe(CAT, num_inference_steps=INV_STEPS, generator=gen(seed)).images[0], f)
    log.time('ddim_inversion', t.s)


LEDITS = {'sleep': ('sleeping cat, closed eyes', False), 'fox': ('a fox', False),
          'paint': ('watercolour painting', False), 'stripes': ('stripes, tabby', True)}
LEDITS_SETTINGS = [(5.0, 0.9), (10.0, 0.75), (15.0, 0.6)]   # (edit guidance scale, threshold): default to strong


@run
def ledits():
    from diffusers import LEditsPPPipelineStableDiffusion
    files = [INV / 'ledits-none.jpg'] + [INV / f'ledits-{k}-g{g}-th{th}.jpg' for k in LEDITS
                                         for g, th in LEDITS_SETTINGS]
    if all(f.exists() for f in files):
        return
    pipe = sd_pipe(LEditsPPPipelineStableDiffusion)
    with Timer() as t:
        pipe.invert(image=inp('cat.png'), source_prompt='', num_inversion_steps=INV_STEPS, skip=0.15,
                    generator=gen(0))
    log.time('leditspp_invert', t.s)
    info['ledits'] = dict(LEDITS, settings=LEDITS_SETTINGS, skip=0.15, scheduler=type(pipe.scheduler).__name__)
    save_info()
    jobs = [('none', None, None, None)] + [(k, e, g, th) for k, e in LEDITS.items() for g, th in LEDITS_SETTINGS]
    for key, e, g, th in jobs:
        f = INV / (f'ledits-{key}.jpg' if e is None else f'ledits-{key}-g{g}-th{th}.jpg')
        if f.exists():
            continue
        kw = {} if e is None else dict(editing_prompt=[e[0]], reverse_editing_direction=[e[1]],
                                       edit_guidance_scale=[g], edit_threshold=[th])
        with Timer() as t:
            save_img(pipe(generator=gen(0), **kw).images[0], f)
        log.time(f'leditspp_{f.stem}', t.s)


# ---------------------------------------------------------------- InstructPix2Pix
S_I = [1.0, 1.25, 1.5, 2.0, 3.0]
S_T = [2.0, 5.0, 7.5, 10.0, 15.0]   # s_T = 1 would switch guidance off altogether in diffusers
TASKS = [('sketch', 'sketch.png', 'make it a photograph'),
         ('rocket', 'rocket.png', 'make it a snowy winter night'),
         ('cat', 'cat.png', 'turn the cat into a bronze statue')]


@run
def pix2pix():
    from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
    pipe = sd_pipe(StableDiffusionInstructPix2PixPipeline, model=PIX2PIX)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    info['pix2pix'] = {'s_I': S_I, 's_T': S_T, 'tasks': {k: i for k, _, i in TASKS}, 'steps': P2P_STEPS,
                       'unet_in_channels': pipe.unet.config.in_channels}
    save_info()
    with Timer() as t:
        n = 0
        for key, file, instr in TASKS:
            image = inp(file)
            for si in S_I:
                for st in S_T:
                    f = OUT / 'pix2pix' / key / f'sI{si}-sT{st}.jpg'
                    if f.exists():
                        continue
                    im = pipe(instr, image=image, num_inference_steps=P2P_STEPS, image_guidance_scale=si,
                              guidance_scale=st, generator=gen(0)).images[0]
                    save_img(im, f)
                    n += 1
    if n:
        log.time('pix2pix', t.s, n)


finish('lec21')

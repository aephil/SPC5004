"""lec20: image-to-image strength sweeps, and inpainting two ways. ~10-20 min on a mid-range GPU.

results/lec20/img2img/<src>/s<strength>-<seed>.jpg   output
results/lec20/img2img/<src>/start-s<strength>.jpg    the noised starting latent, decoded (what denoising starts from)
results/lec20/img2img/<src>/t2i-<seed>.jpg           plain text-to-image with the same prompt and seed
results/lec20/inpaint/<method>-<mask>-s<strength>-<seed>.jpg   method: blend (ordinary SD v1.5) or model (inpainting UNet)
results/lec20/inpaint/traj-<method>/<i>.jpg          latents during one inpainting run, decoded every few steps
Each section is independent: one failing doesn't stop the others.
"""
import torch

from common import (BASE, INPAINT, NEGATIVE, RESULTS, STEPS, Log, Timer, finish, gen, inp, run, save_img, save_json,
                    sd_pipe)

log = Log('lec20')
OUT = RESULTS / 'lec20'
STRENGTHS = [round(0.1 * i, 1) for i in range(1, 11)]
STRENGTHS = [s for s in STRENGTHS if int(STEPS * s) >= 1]       # (only matters in quick mode)
SEEDS = [1, 2]
SOURCES = [
    ('sketch', 'sketch.png', 'a photograph of a stone cottage on a green hill with a tree, sunny day, detailed'),
    ('cat', 'cat.png', 'a photo of a red fox, close-up'),
]


@torch.no_grad()
def decode(pipe, z):
    x = pipe.vae.decode(z / pipe.vae.config.scaling_factor).sample
    return pipe.image_processor.postprocess(x, output_type='pil')[0]


@run
def img2img():
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe = sd_pipe(StableDiffusionImg2ImgPipeline)
    info = {'steps': STEPS, 'strengths': STRENGTHS, 'sources': {k: p for k, _, p in SOURCES}, 'first_timestep': {}}
    with Timer() as t:
        n = 0
        for key, file, prompt in SOURCES:
            image = inp(file)
            with torch.no_grad():
                z0 = pipe.vae.encode(pipe.image_processor.preprocess(image).to('cuda', torch.float16)).latent_dist.mean
            z0 = z0 * pipe.vae.config.scaling_factor
            for s in STRENGTHS:
                # What the loop starts from at this strength (same noise for every strength, for illustration)
                pipe.scheduler.set_timesteps(STEPS)
                ts, n_run = pipe.get_timesteps(STEPS, s, 'cuda')
                info['first_timestep'][str(s)] = {'t': int(ts[0]), 'steps_run': int(n_run)}
                f = OUT / 'img2img' / key / f'start-s{s}.jpg'
                if not f.exists():
                    noise = torch.randn(z0.shape, generator=gen(0), device='cuda', dtype=z0.dtype)
                    save_img(decode(pipe, pipe.scheduler.add_noise(z0, noise, ts[:1])), f)
                for seed in SEEDS:
                    f = OUT / 'img2img' / key / f's{s}-{seed}.jpg'
                    if f.exists():
                        continue
                    im = pipe(prompt, image=image, strength=s, negative_prompt=NEGATIVE, num_inference_steps=STEPS,
                              guidance_scale=7.5, generator=gen(seed)).images[0]
                    save_img(im, f)
                    n += 1
    if n:
        log.time('img2img', t.s, n)
    info['alphabar_first_t'] = float(pipe.scheduler.alphas_cumprod[info['first_timestep'][str(STRENGTHS[-1])]['t']])
    save_json(info, OUT / 'img2img.json')


@run
def text_to_image():
    """The same prompts and seeds from pure noise, to set beside strength 1.0."""
    pipe = sd_pipe()
    for key, _, prompt in SOURCES:
        for seed in SEEDS:
            f = OUT / 'img2img' / key / f't2i-{seed}.jpg'
            if not f.exists():
                save_img(pipe(prompt, negative_prompt=NEGATIVE, num_inference_steps=STEPS,
                              generator=gen(seed)).images[0], f)


PROMPT = 'a small potted cactus on a saucer, wooden table'
ip_info = {'prompt': PROMPT}


def inpaint(method, model):
    from diffusers import StableDiffusionInpaintPipeline
    image = inp('coffee.png')
    masks = {'tight': inp('coffee-mask-tight.png'), 'loose': inp('coffee-mask-loose.png')}
    pipe = sd_pipe(StableDiffusionInpaintPipeline, model=model)
    ip_info[f'{method}_unet_in_channels'] = pipe.unet.config.in_channels
    save_json(ip_info, OUT / 'inpaint.json')
    with Timer() as t:
        n = 0
        for mk, m in masks.items():
            for s in [0.75, 1.0]:
                for seed in SEEDS:
                    f = OUT / 'inpaint' / f'{method}-{mk}-s{s}-{seed}.jpg'
                    if f.exists():
                        continue
                    im = pipe(PROMPT, image=image, mask_image=m, strength=s, negative_prompt=NEGATIVE,
                              num_inference_steps=STEPS, generator=gen(seed)).images[0]
                    save_img(im, f)
                    n += 1
    if n:
        log.time(f'inpaint_{method}', t.s, n)

    # One run with its latents decoded every 3 steps: for the blend, shows the kept region put back each step
    traj = OUT / 'inpaint' / f'traj-{method}'
    if not (traj / f'{STEPS - 1:02d}.jpg').exists():
        def grab(p, i, t, kw):
            if i % 3 == 2 or i == STEPS - 1:
                save_img(decode(p, kw['latents']), traj / f'{i:02d}.jpg')
            return {}
        pipe(PROMPT, image=image, mask_image=masks['tight'], strength=1.0, negative_prompt=NEGATIVE,
             num_inference_steps=STEPS, generator=gen(1), callback_on_step_end=grab,
             callback_on_step_end_tensor_inputs=['latents'])


@run
def inpaint_blend():
    inpaint('blend', BASE)            # ordinary SD v1.5: 4-channel UNet, known region pasted back every step


@run
def inpaint_model():
    inpaint('model', INPAINT)         # SD v1.5 inpainting: 9-channel UNet, sees the mask and the masked image


finish('lec20')

"""lec22: IP-Adapter and ControlNet, separately and together. ~15-25 min on a mid-range GPU.

results/lec22/ip/<prompt>-<ref>-s<scale>-<seed>.jpg     image prompt at several scales (s0 = text only)
results/lec22/ip/<prompt>-text<ref>-<seed>.jpg           the style asked for in words instead
results/lec22/ip/variation-<ref>-<seed>.jpg              image prompt only (empty text)
results/lec22/cn/<cond>.png                              the control images (edges, depth, pose)
results/lec22/cn/<cond>-<prompt>-c<scale>.jpg
results/lec22/both/knight-<ref>-ip<..>-c0.8.jpg           depth ControlNet + IP-Adapter + text
results/lec22/info.json                                  parameter counts and settings
Each section is independent: one failing doesn't stop the others.
"""
import numpy as np
import torch
from PIL import Image

from common import (CONTROLNETS, DEPTH_ESTIMATOR, IP_ADAPTER, NEGATIVE, RESULTS, STEPS, Log, Timer, finish, gen, inp,
                    load_json, run, save_img, save_json, sd_pipe)

log = Log('lec22')
OUT = RESULTS / 'lec22'
info = load_json(OUT / 'info.json', {})
REFS = {'hiroshige': inp('hiroshige.png'), 'vangogh': inp('vangogh.png')}
REF_WORDS = {'hiroshige': 'Japanese woodblock print', 'vangogh': 'painting by Vincent van Gogh'}
IP_PROMPTS = {'bicycle': 'a red bicycle leaning against a yellow wall', 'cat': 'a cat sleeping on a cushion'}
IP_SCALES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
PHOTO = inp('astronaut.png')
CN_PROMPTS = {'knight': 'an oil painting of a knight in armour, castle in the background',
              'robot': 'a photograph of a robot in a workshop', 'none': ''}
CN_SCALES = [0.0, 0.3, 0.6, 1.0, 1.5]


def save_info():
    save_json(info, OUT / 'info.json')


def count(params):
    return int(sum(p.numel() for p in params))


@run
def ip_adapter():
    pipe = sd_pipe()
    unet_params = count(pipe.unet.parameters())
    pipe.load_ip_adapter(*IP_ADAPTER)
    info['ip_adapter'] = {
        'unet_params_before': unet_params,
        'new_kv_params': count(p for n, p in pipe.unet.named_parameters() if 'to_k_ip' in n or 'to_v_ip' in n),
        'projection_params': count(pipe.unet.encoder_hid_proj.parameters()),
        'image_encoder_params': count(pipe.image_encoder.parameters()),
        'image_encoder': getattr(pipe.image_encoder.config, '_name_or_path', ''),
        'n_cross_attention_layers': sum(1 for n, _ in pipe.unet.named_modules() if n.endswith('attn2')),
        'scales': IP_SCALES, 'prompts': IP_PROMPTS, 'ref_words': REF_WORDS,
    }
    try:                                   # shapes only, for the slides; never worth stopping for
        with torch.no_grad():
            emb = pipe.prepare_ip_adapter_image_embeds([REFS['vangogh']], None, 'cuda', 1, True)
            info['ip_adapter']['image_embeds_shape'] = list(emb[0].shape)
            tokens = pipe.unet.encoder_hid_proj.image_projection_layers[0](emb[0][-1:].to(torch.float16))
            info['ip_adapter']['image_tokens_shape'] = list(tokens.shape)
    except Exception as e:                 # noqa: BLE001
        print('shape probe failed:', repr(e)[:200])
    save_info()
    print(info['ip_adapter'])

    with Timer() as t:
        n = 0
        for pk, prompt in IP_PROMPTS.items():
            for rk, ref in REFS.items():
                for s in IP_SCALES:
                    f = OUT / 'ip' / f'{pk}-{rk}-s{s}-11.jpg'
                    if f.exists():
                        continue
                    pipe.set_ip_adapter_scale(s)
                    im = pipe(prompt, ip_adapter_image=ref, negative_prompt=NEGATIVE, num_inference_steps=STEPS,
                              generator=gen(11)).images[0]
                    save_img(im, f)
                    n += 1
                f = OUT / 'ip' / f'{pk}-text{rk}-11.jpg'
                if not f.exists():
                    pipe.set_ip_adapter_scale(0.0)
                    im = pipe(f'{prompt}, {REF_WORDS[rk]}', ip_adapter_image=ref, negative_prompt=NEGATIVE,
                              num_inference_steps=STEPS, generator=gen(11)).images[0]
                    save_img(im, f)
                    n += 1
        pipe.set_ip_adapter_scale(1.0)
        for rk, ref in REFS.items():
            for seed in [1, 2]:
                f = OUT / 'ip' / f'variation-{rk}-{seed}.jpg'
                if not f.exists():
                    save_img(pipe('', ip_adapter_image=ref, negative_prompt=NEGATIVE, num_inference_steps=STEPS,
                                  generator=gen(seed)).images[0], f)
                    n += 1
    if n:
        log.time('ip_adapter', t.s, n)


# ---------------------------------------------------------------- control images from one photo
@run
def canny():
    f = OUT / 'cn' / 'canny.png'
    if not f.exists():
        import cv2
        edges = cv2.Canny(np.array(PHOTO), 100, 200)
        save_img(Image.fromarray(np.stack([edges] * 3, -1)), f)


@run
def depth():
    f = OUT / 'cn' / 'depth.png'
    if not f.exists():
        from transformers import pipeline as hf_pipeline
        d = hf_pipeline('depth-estimation', model=DEPTH_ESTIMATOR, device=0)(PHOTO)['depth']
        d = np.array(d.resize(PHOTO.size)).astype(np.float32)
        d = (255 * (d - d.min()) / (d.max() - d.min() + 1e-6)).astype(np.uint8)
        save_img(Image.fromarray(np.stack([d] * 3, -1)), f)


@run
def pose():
    f = OUT / 'cn' / 'pose.png'
    if not f.exists():
        try:
            from controlnet_aux import OpenposeDetector
        except ImportError:
            print('controlnet_aux not installed: skipping the pose example')
            return
        save_img(OpenposeDetector.from_pretrained('lllyasviel/Annotators')(PHOTO).resize(PHOTO.size), f)


# ---------------------------------------------------------------- ControlNet, one section per kind of map
def controlnet(ck):
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    cimg = Image.open(OUT / 'cn' / f'{ck}.png').convert('RGB')
    cn = ControlNetModel.from_pretrained(CONTROLNETS[ck], torch_dtype=torch.float16)
    info['controlnet'] = dict(info.get('controlnet', {}), prompts=CN_PROMPTS, scales=CN_SCALES, models=CONTROLNETS,
                              params=count(cn.parameters()),
                              cond_embedding_params=count(cn.controlnet_cond_embedding.parameters()),
                              n_zero_convs=len(cn.controlnet_down_blocks) + 1)
    save_info()
    pipe = sd_pipe(StableDiffusionControlNetPipeline, controlnet=cn)
    jobs = [('knight', s) for s in CN_SCALES] + [('robot', 1.0), ('none', 1.0)]
    with Timer() as t:
        n = 0
        for pk, s in jobs:
            f = OUT / 'cn' / f'{ck}-{pk}-c{s}.jpg'
            if f.exists():
                continue
            im = pipe(CN_PROMPTS[pk], image=cimg, controlnet_conditioning_scale=s, negative_prompt=NEGATIVE,
                      num_inference_steps=STEPS, generator=gen(5)).images[0]
            save_img(im, f)
            n += 1
    if n:
        log.time(f'controlnet_{ck}', t.s, n)

    if ck == 'depth':                       # everything at once: depth map + style image + text
        pipe.load_ip_adapter(*IP_ADAPTER)
        for rk, ref in REFS.items():
            for ips in [0.0, 0.5, 0.7]:
                f = OUT / 'both' / f'knight-{rk}-ip{ips}-c0.8.jpg'
                if f.exists():
                    continue
                pipe.set_ip_adapter_scale(ips)
                im = pipe('a knight in armour', image=cimg, ip_adapter_image=ref, controlnet_conditioning_scale=0.8,
                          negative_prompt=NEGATIVE, num_inference_steps=STEPS, generator=gen(5)).images[0]
                save_img(im, f)


@run
def controlnet_canny():
    controlnet('canny')


@run
def controlnet_depth():
    controlnet('depth')


@run
def controlnet_pose():
    if (OUT / 'cn' / 'pose.png').exists():
        controlnet('pose')


finish('lec22')

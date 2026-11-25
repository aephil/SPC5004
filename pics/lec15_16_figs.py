"""Make the images and measured numbers used in lec15.qmd and lec16.qmd.

Needs an env with torch, diffusers, scikit-image and Pillow (the genai kernel lacks diffusers). Run from
the slides folder:

    python pics/lec15_16_figs.py numbers            # parameter counts, schedule values -> stdout
    python pics/lec15_16_figs.py forward            # pics/lec15/forward-*.jpg, pics/lec16/forward-res.jpg
    python pics/lec15_16_figs.py reverse            # pics/lec15/reverse.jpg (google/ddpm-cat-256, ~455 MB)
    python pics/lec15_16_figs.py cond CKPT_DIR      # pics/lec16/*.png, from pics/lec16_train.py's model

The photo is scikit-image's `chelsea` (CC0). Each image strip is one JPEG/PNG with thin white gaps, so
the .qmd can show it with a plain ![](...).
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
DEV = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def strip(images, size, gap=6, path=None, rows=1):
    """Tile a list of HxWxC uint8 arrays (row-major) into one image with white gaps."""
    cols = len(images) // rows
    out = Image.new('RGB', (cols * size + (cols - 1) * gap, rows * size + (rows - 1) * gap), 'white')
    for k, im in enumerate(images):
        r, c = divmod(k, cols)
        pil = Image.fromarray(im).convert('RGB').resize((size, size), Image.NEAREST if im.shape[0] < size else
                                                        Image.LANCZOS)
        out.paste(pil, (c * (size + gap), r * (size + gap)))
    out.save(path, quality=90) if path.suffix == '.jpg' else out.save(path)
    print('wrote', path, out.size)


def to_uint8(x):
    """CxHxW tensor in [-1, 1] -> HxWxC uint8."""
    x = ((x.clamp(-1, 1) + 1) * 127.5).round().byte().cpu()
    x = x.permute(1, 2, 0).numpy()
    return x[:, :, 0] if x.shape[2] == 1 else x


def cat_photo(res=256):
    from skimage import data
    im = Image.fromarray(data.chelsea())                     # 300 x 451
    w, h = im.size
    im = im.crop(((w - h) // 2 + 30, 0, (w - h) // 2 + 30 + h, h)).resize((res, res), Image.LANCZOS)
    return torch.tensor(np.asarray(im), dtype=torch.float32).permute(2, 0, 1) / 127.5 - 1


def numbers():
    from diffusers import DDPMScheduler, UNet2DModel
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    lab15 = UNet2DModel(sample_size=32, in_channels=1, out_channels=1, layers_per_block=2,
                        block_out_channels=(32, 64, 64),
                        down_block_types=('DownBlock2D', 'AttnDownBlock2D', 'AttnDownBlock2D'),
                        up_block_types=('AttnUpBlock2D', 'AttnUpBlock2D', 'UpBlock2D'))
    lab16 = UNet2DModel(in_channels=1, out_channels=1, sample_size=32, block_out_channels=(32, 64, 128, 256),
                        num_class_embeds=10)
    print('lab15 UNet2DModel params', count(lab15))
    print('lab16 UNet2DModel params', count(lab16), 'class table', tuple(lab16.class_embedding.weight.shape),
          'time_proj', lab16.time_proj.num_channels, 'time_embedding', lab16.time_embedding)
    print('lab16 down blocks', [type(b).__name__ for b in lab16.down_blocks])
    with torch.no_grad():
        x = torch.randn(1, 1, 32, 32)
        print('out shape', lab16(x, 7, class_labels=torch.tensor([2])).sample.shape)
    for name, kw in [('ddpm', dict(beta_start=1e-4, beta_end=0.02)), ('lab15', dict(beta_start=1e-3, beta_end=0.02)),
                     ('cosine', dict(beta_schedule='squaredcos_cap_v2'))]:
        s = DDPMScheduler(num_train_timesteps=1000, **kw)
        ab = s.alphas_cumprod
        ts = [0, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 999]
        print(name, 'beta[0,999]', float(s.betas[0]), float(s.betas[-1]))
        print(name, 'alpha_bar', {t: round(float(ab[t]), 4) for t in ts})
        print(name, 'sqrt alpha_bar', {t: round(float(ab[t]) ** .5, 3) for t in ts})
        print(name, 'first t with sqrt(ab)<0.1', int((ab.sqrt() < 0.1).nonzero()[0]))
        np.savetxt(HERE / 'lec15' / f'alphabar-{name}.txt', ab.numpy()[::10], fmt='%.5f')


def forward():
    from diffusers import DDPMScheduler
    x0 = cat_photo(256)
    g = torch.Generator().manual_seed(1)
    # simplest version: mix with uniform noise, in [0, 1]
    u = torch.rand(x0.shape, generator=g) * 2 - 1
    amts = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    strip([to_uint8(x0 * (1 - a) + u * a) for a in amts], 150, path=HERE / 'lec15' / 'forward-mix.jpg')
    # DDPM: Gaussian noise, linear schedule
    s = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    eps = torch.randn(x0.shape, generator=g)
    ts = [0, 50, 100, 200, 400, 600, 999]
    xs = s.add_noise(x0[None].repeat(len(ts), 1, 1, 1), eps[None].repeat(len(ts), 1, 1, 1), torch.tensor(ts))
    strip([to_uint8(x) for x in xs], 130, path=HERE / 'lec15' / 'forward-ddpm.jpg')
    # the same noise level at two resolutions
    small = torch.nn.functional.interpolate(x0[None], size=32, mode='area')[0]
    eps32 = torch.randn(small.shape, generator=g)
    ts2 = [0, 100, 200, 300]
    row = []
    for t in ts2:
        tt = torch.tensor([t])
        row.append(to_uint8(s.add_noise(x0[None], eps[None], tt)[0]))
    for t in ts2:
        tt = torch.tensor([t])
        row.append(to_uint8(s.add_noise(small[None], eps32[None], tt)[0]))
    strip(row, 130, path=HERE / 'lec16' / 'forward-res.jpg', rows=2)


@torch.no_grad()
def reverse():
    from diffusers import DDPMScheduler, UNet2DModel
    unet = UNet2DModel.from_pretrained('google/ddpm-cat-256').to(DEV)
    s = DDPMScheduler.from_pretrained('google/ddpm-cat-256')
    print('cat-256 params', sum(p.numel() for p in unet.parameters()), 'schedule', s.config.beta_schedule,
          s.config.beta_start, s.config.beta_end, 'prediction', s.config.prediction_type)
    s.set_timesteps(1000)
    torch.manual_seed(4)
    n = 4  # lec15 uses samples 0 and 1
    x = torch.randn(n, 3, 256, 256).to(DEV)
    keep = [999, 900, 800, 700, 600, 400, 200, 0]
    xt, x0 = {}, {}
    for t in s.timesteps:
        eps = unet(x, t).sample
        out = s.step(eps, t, x)
        if int(t) in keep:
            xt[int(t)], x0[int(t)] = x.cpu(), out.pred_original_sample.cpu()
        x = out.prev_sample
        if int(t) % 100 == 0:
            print('t', int(t), flush=True)
    for i in range(n):
        strip([to_uint8(xt[t][i]) for t in keep] + [to_uint8(x0[t][i]) for t in keep], 120,
              path=HERE / 'lec15' / f'reverse-{i}.jpg', rows=2)
    strip([to_uint8(x[i].cpu()) for i in range(n)], 200, path=HERE / 'lec15' / 'cats.jpg')


@torch.no_grad()
def cond(ckpt):
    from diffusers import DDPMScheduler, UNet2DModel
    unet = UNet2DModel.from_pretrained(ckpt).to(DEV).eval()
    s = DDPMScheduler(num_train_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    s.set_timesteps(1000)

    import diffusers.schedulers.scheduling_ddpm as ddpm_mod
    from diffusers.utils.torch_utils import randn_tensor

    def run(noise, labels=None, emb=None, seed=0, shared=True):
        """DDPM sampling. With `shared`, every image in the batch gets the same fresh noise at each step
        (as well as the same start, if `noise` repeats), so only the label differs between them."""
        gen = torch.Generator(device='cpu').manual_seed(seed)

        def fresh(shape, generator=None, device=None, dtype=None, **kw):
            one = randn_tensor((1, *shape[1:]) if shared else shape, generator=gen, dtype=dtype)
            return one.expand(shape).to(device) if shared else one.to(device)

        ddpm_mod.randn_tensor = fresh
        x = noise.clone()
        for t in s.timesteps:
            if emb is not None:      # swap in a hand-made class vector
                unet.class_embedding.weight.data[0] = emb
            x = s.step(unet(x, t, class_labels=labels).sample, t, x).prev_sample
        ddpm_mod.randn_tensor = randn_tensor
        return x

    # rows: 3 starting noises; columns: the ten labels, same noise across a row
    torch.manual_seed(0)
    seeds = torch.randn(3, 1, 1, 32, 32)
    imgs = []
    for r in range(3):
        noise = seeds[r].repeat(10, 1, 1, 1).to(DEV)
        out = run(noise, torch.arange(10, device=DEV), seed=r).cpu()
        imgs += [to_uint8(o) for o in out]
    strip(imgs, 96, gap=4, path=HERE / 'lec16' / 'same-noise.png', rows=3)
    # one label, eight noises
    noise = torch.randn(8, 1, 32, 32).to(DEV)
    out = run(noise, torch.full((8,), 7, device=DEV), seed=10, shared=False).cpu()
    strip([to_uint8(o) for o in out], 96, gap=4, path=HERE / 'lec16' / 'sevens.png')
    # blends of two class vectors
    table = unet.class_embedding.weight.data.clone()
    noise = seeds[1].to(DEV)
    blend = []
    for w in [0, 0.25, 0.5, 0.75, 1]:
        unet.class_embedding.weight.data.copy_(table)
        e = (1 - w) * table[1] + w * table[7]
        blend.append(to_uint8(run(noise, torch.zeros(1, dtype=torch.long, device=DEV), emb=e, seed=1)[0].cpu()))
    unet.class_embedding.weight.data.copy_(table)
    strip(blend, 96, gap=4, path=HERE / 'lec16' / 'blend.png')
    print('class table norms', table.norm(dim=1).cpu().numpy().round(2))


if __name__ == '__main__':
    {'numbers': numbers, 'forward': forward, 'reverse': reverse,
     'cond': lambda: cond(sys.argv[2])}[sys.argv[1]]()

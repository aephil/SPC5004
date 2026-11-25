"""Settings and helpers shared by the lec19-22 GPU jobs. Edit settings here, not in the job scripts."""
import json
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUTS = HERE / 'inputs'
QUICK = bool(os.environ.get('L19_QUICK'))  # smoke test: every code path, few steps, separate folders
WORK = Path(os.environ.get('L19_WORK', HERE / ('work-quick' if QUICK else 'work')))           # big files
RESULTS = Path(os.environ.get('L19_RESULTS', HERE / ('results-quick' if QUICK else 'results')))  # sent back

BASE = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
INPAINT = 'stable-diffusion-v1-5/stable-diffusion-inpainting'
PIX2PIX = 'timbrooks/instruct-pix2pix'
IP_ADAPTER = ('h94/IP-Adapter', 'models', 'ip-adapter_sd15.bin')
CONTROLNETS = {'canny': 'lllyasviel/control_v11p_sd15_canny',
               'depth': 'lllyasviel/control_v11f1p_sd15_depth',
               'pose': 'lllyasviel/control_v11p_sd15_openpose'}
DEPTH_ESTIMATOR = 'Intel/dpt-hybrid-midas'
DINO = 'facebook/dino-vits16'
CLIP = 'openai/clip-vit-large-patch14'

STEPS = 30          # sampling steps everywhere unless a job says otherwise
GUIDANCE = 7.5
NEGATIVE = 'low quality, blurry, deformed'

# ---- lec19: DreamBooth (LoRA) on the DreamBooth dataset's "dog" (Unsplash, @alvannee)
INSTANCE_PROMPT = 'a photo of sks dog'
CLASS_PROMPT = 'a photo of a dog'
N_CLASS = 200
DB_STEPS = int(os.environ.get('L19_STEPS', 1600))
DB_CHECKPOINT_EVERY = 100
DB_SNAPSHOTS = [0, 100, 200, 400, 800, 1200, 1600]
DB_SETTINGS = dict(learning_rate=1e-4, rank=8, train_text_encoder=True)   # LoRA on the UNet and text encoder
DB_RUNS = ['prior', 'noprior']
DB_PROMPTS = [
    ('subject', 'a photo of sks dog'),
    ('beach', 'a photo of sks dog running on a beach'),
    ('hat', 'a photo of sks dog wearing a red hat'),
    ('paint', 'a watercolour painting of sks dog'),
    ('bucket', 'a photo of sks dog in a bucket'),
    ('dog', 'a photo of a dog'),
    ('park', 'a dog playing in a park'),
    ('cat', 'a photo of a cat'),
]
DB_SEEDS = [1, 2]
DB_DRIFT_SEEDS = list(range(100, 108))   # extra "a photo of a dog" samples, for measuring drift
DB_STEP0_EXTRA = [('sks', 'a photo of sks'), ('corgi', 'a photo of a corgi')]


if QUICK:
    STEPS, N_CLASS, DB_STEPS, DB_CHECKPOINT_EVERY, DB_SNAPSHOTS = 4, 4, 20, 10, [0, 10, 20]
    DB_DRIFT_SEEDS = DB_DRIFT_SEEDS[:2]


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No GPU visible to PyTorch (torch.cuda.is_available() is False).')
    p = torch.cuda.get_device_properties(0)
    return {'name': p.name, 'vram_gb': round(p.total_memory / 2**30, 2), 'cuda': torch.version.cuda,
            'torch': torch.__version__}


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def save_img(im, path, quality=92):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + '.tmp' + path.suffix)
    if path.suffix == '.jpg':
        im.convert('RGB').save(tmp, quality=quality)
    else:
        im.save(tmp)
    tmp.replace(path)


def gen(seed):
    import torch
    return torch.Generator('cuda').manual_seed(seed)


def free():
    """Call after `del pipe` to give the memory back before loading the next model."""
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()


class Log:
    """Appends timing notes to results/<name>-log.json, so the slides can quote real speeds."""

    def __init__(self, name):
        self.path = RESULTS / f'{name}-log.json'
        self.data = load_json(self.path, {})

    def time(self, key, seconds, n=1):
        self.data[key] = {'seconds': round(seconds, 2), 'n': n, 'per_item': round(seconds / max(n, 1), 3)}
        save_json(self.data, self.path)

    def note(self, key, value):
        self.data[key] = value
        save_json(self.data, self.path)


class Timer:
    def __enter__(self):
        import torch
        torch.cuda.synchronize()
        self.t = time.time()
        return self

    def __exit__(self, *a):
        import torch
        torch.cuda.synchronize()
        self.s = time.time() - self.t


def sd_pipe(cls=None, model=BASE, **kw):
    """A Stable Diffusion pipeline in fp16 on the GPU, without the safety checker (outputs are reviewed by hand)."""
    import torch
    from diffusers import StableDiffusionPipeline
    cls = cls or StableDiffusionPipeline
    extra = {'safety_checker': None, 'requires_safety_checker': False}
    try:
        p = cls.from_pretrained(model, torch_dtype=torch.float16, variant='fp16', **extra, **kw)
    except (OSError, ValueError):
        p = cls.from_pretrained(model, torch_dtype=torch.float16, **extra, **kw)
    p.set_progress_bar_config(disable=True)
    return p.to('cuda')


def inp(name, size=512):
    from PIL import Image
    im = Image.open(INPUTS / name)
    return im.convert('L' if 'mask' in name else 'RGB').resize((size, size))


FAILED = []


def run(fn):
    """Run one independent section of a job. If it fails, write the traceback to results/errors.txt, free the
    GPU and carry on with the next section; finish() then exits non-zero so run_all.sh reports the failure."""
    import traceback
    print(f'--- {fn.__name__}', flush=True)
    try:
        fn()
    except Exception:                                          # noqa: BLE001
        tb = traceback.format_exc()
        print(tb, flush=True)
        RESULTS.mkdir(parents=True, exist_ok=True)
        with open(RESULTS / 'errors.txt', 'a') as f:
            f.write(f'==== {fn.__module__}.{fn.__name__}\n{tb}\n')
        FAILED.append(fn.__name__)
    free()
    return fn


def finish(job):
    import sys
    if FAILED:
        sys.exit(f'{job}: failed sections: {", ".join(FAILED)} (see errors.txt)')
    print(f'{job} done')

"""Settings shared by the lec18 GPU scripts. Edit here, not in the individual scripts."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = Path(os.environ.get('LEC18_WORK', HERE / 'work'))       # big files: images, checkpoints (not returned)
RESULTS = Path(os.environ.get('LEC18_RESULTS', HERE / 'results'))  # small files: zipped and sent back

DATA = WORK / 'data'               # imagefolder: DATA/train/*.jpg + metadata.jsonl; DATA/heldout/...
RUN = WORK / 'run'                 # output_dir of train_text_to_image.py
BASE = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
CAPTIONER = 'Salesforce/blip-image-captioning-large'
CLIP = 'openai/clip-vit-large-patch14'

N_HELDOUT = 40                     # prints kept out of training, for the held-out loss
MAX_ASPECT = 2.0                   # skip very tall or wide prints: a 512 x 512 centre crop would lose most of them
SHORT_SIDE = 768                   # stored size; the training script resizes to 512 itself

# Training: the lab's recipe, with an effective batch of 4 however it is split.
EFFECTIVE_BATCH = 4
LEARNING_RATE = 1e-5
MIXED_PRECISION = os.environ.get('LEC18_MP', 'fp16')
# Snapshots, roughly doubling, so that a run cut short still gives a usable sequence.
STAGES = [int(s) for s in os.environ.get('LEC18_STAGES', '50,100,200,400,800,1600,3200,6400').split(',')]

# Sampling: same prompts, seeds and settings at every stage. Three prompts are lec17's.
PROMPTS = [
    ('lighthouse', 'a lighthouse on a rocky cliff above the sea at sunset, oil painting'),
    ('bicycle', 'a red bicycle leaning against a yellow wall'),
    ('ramen', 'a bowl of ramen with chopsticks, photograph'),
    ('astronaut', 'a portrait photograph of an astronaut in a white spacesuit'),
    ('cat', 'a cat sleeping on a cushion'),
    ('bridge', 'people crossing a wooden bridge in the rain'),
    ('kimono', 'a woman in a kimono holding a fan'),
]
N_TRAIN_PROBES = 2                 # plus this many prompts copied from training captions (memorisation check)
SEEDS = [11, 22]
SAMPLE_STEPS = 30
GUIDANCE = 7.5

# Fixed-noise loss evaluation at each stage.
EVAL_TIMESTEPS = [50, 200, 400, 600, 800, 950]
EVAL_N_TRAIN = 40


def gpu_info():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('No GPU visible to PyTorch (torch.cuda.is_available() is False). '
                         'On AMD, install the ROCm build of PyTorch.')
    p = torch.cuda.get_device_properties(0)
    return {'name': p.name, 'vram_gb': round(p.total_memory / 2**30, 1),
            'hip': getattr(torch.version, 'hip', None), 'cuda': torch.version.cuda, 'torch': torch.__version__}


def recipe(vram_gb, bnb_ok):
    """Training settings for this card. Always fp16 mixed precision, EMA and an effective batch of 4.
    Very large cards (>= 60 GB): micro-batch 4, plain AdamW, no gradient checkpointing (fastest).
    Large (>= 40 GB): micro-batch 4, plain AdamW, gradient checkpointing.
    Mid-size (>= 22 GB): micro-batch 2 x 2, checkpointing, 8-bit Adam if available.
    Smaller: micro-batch 1 x 4, checkpointing, 8-bit Adam if available (may still not fit below ~16 GB).
    LEC18_BS overrides the micro-batch."""
    if vram_gb >= 60:
        bs, ckpt, eight = 4, False, False
    elif vram_gb >= 40:
        bs, ckpt, eight = 4, True, False
    elif vram_gb >= 22:
        bs, ckpt, eight = 2, True, bnb_ok
    else:
        bs, ckpt, eight = 1, True, bnb_ok
    if os.environ.get('LEC18_BS'):
        bs = int(os.environ['LEC18_BS'])
    return {'train_batch_size': bs, 'gradient_accumulation_steps': EFFECTIVE_BATCH // bs,
            'gradient_checkpointing': ckpt, 'use_8bit_adam': eight, 'use_ema': True,
            'mixed_precision': MIXED_PRECISION, 'learning_rate': LEARNING_RATE}


def have_bnb():
    """True if bitsandbytes' 8-bit AdamW actually runs on this GPU (ROCm support varies)."""
    try:
        import bitsandbytes as bnb
        import torch
        p = torch.nn.Parameter(torch.randn(4096, device='cuda'))
        opt = bnb.optim.AdamW8bit([p], lr=1e-3)
        p.grad = torch.randn_like(p)
        opt.step()
        return bool(torch.isfinite(p).all())
    except Exception as e:                      # noqa: BLE001 - any failure means "don't use it"
        print('bitsandbytes 8-bit Adam unavailable:', repr(e)[:200])
        return False


def probes():
    """All prompts as (key, text): the fixed ones plus captions of the first N_TRAIN_PROBES training prints."""
    meta = [json.loads(l) for l in (DATA / 'train' / 'metadata.jsonl').read_text().splitlines()]
    meta.sort(key=lambda m: m['file_name'])
    return PROMPTS + [(f'train{k}', m['text']) for k, m in enumerate(meta[:N_TRAIN_PROBES])]


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)

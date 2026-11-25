"""lec19: measurements for the slides.

- DINO (ViT-S/16) and CLIP image embeddings of every sample, the five subject photos and the class images.
  Writes results/lec19/similarity.json: for each image, mean cosine similarity to the subject photos, and,
  for each (run, step), how similar the "a photo of a dog" samples are to the subject and to each other.
- The logged training loss of each run (from tensorboard) -> results/lec19/loss-<run>.csv
- LoRA sizes: trainable parameters and file size -> results/lec19/lora.json
"""
import csv
import itertools

import torch
from PIL import Image

from common import CLIP, DB_RUNS, DINO, INPUTS, RESULTS, WORK, save_json

OUT = RESULTS / 'lec19'
SAMPLES = OUT / 'samples'


def embed(name, paths, device='cuda'):
    """Unit-length image embeddings. Falls back to the CPU if the GPU runs out of memory."""
    from transformers import AutoImageProcessor, AutoModel, CLIPModel
    dtype = torch.float16 if device == 'cuda' else torch.float32
    proc = AutoImageProcessor.from_pretrained(name)
    if 'clip' in name:
        m = CLIPModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        f = lambda x: m.get_image_features(pixel_values=x)                     # noqa: E731
    else:
        m = AutoModel.from_pretrained(name, torch_dtype=dtype).to(device).eval()
        f = lambda x: m(pixel_values=x).last_hidden_state[:, 0]                # CLS token, as in the paper # noqa
    try:
        out = []
        with torch.no_grad():
            for i in range(0, len(paths), 8):
                ims = [Image.open(p).convert('RGB') for p in paths[i:i + 8]]
                x = proc(images=ims, return_tensors='pt')['pixel_values'].to(device, dtype)
                out.append(torch.nn.functional.normalize(f(x).float(), dim=-1).cpu())
        return torch.cat(out)
    except torch.cuda.OutOfMemoryError:
        if device != 'cuda':
            raise
        print('GPU out of memory: embedding on the CPU instead', flush=True)
        torch.cuda.empty_cache()
        return embed(name, paths, 'cpu')


if torch.cuda.is_available():
    free_b, total_b = torch.cuda.mem_get_info()
    print(f'GPU memory free at start: {free_b / 2**30:.2f} of {total_b / 2**30:.2f} GiB', flush=True)


subject = sorted((INPUTS / 'dog').glob('*.jpg'))
klass = sorted((WORK / 'class').glob('*.jpg'))[:60]
samples = sorted(SAMPLES.rglob('*.jpg'))
paths = subject + klass + samples
res = {'subject': [p.name for p in subject], 'n_class': len(klass)}
for name, key in [(DINO, 'dino'), (CLIP, 'clip')]:
    E = embed(name, paths)
    torch.cuda.empty_cache()
    S, C, X = E[:len(subject)], E[len(subject):len(subject) + len(klass)], E[len(subject) + len(klass):]
    res[f'{key}_subject_self'] = round(float((S @ S.T)[~torch.eye(len(S), dtype=bool)].mean()), 4)
    res[f'{key}_class_to_subject'] = round(float((C @ S.T).mean()), 4)
    res[f'{key}_class_self'] = round(float((C @ C.T)[~torch.eye(len(C), dtype=bool)].mean()), 4)
    per = {}
    for p, x in zip(samples, X):
        per[str(p.relative_to(SAMPLES))] = round(float((x @ S.T).mean()), 4)
    res[f'{key}_per_image'] = per
    # drift: "a photo of a dog" samples (keys dog and dogx), per (run, step)
    groups = {}
    for p, x in zip(samples, X):
        if p.name.split('-')[0] in ('dog', 'dogx'):
            groups.setdefault(str(p.parent.relative_to(SAMPLES)), []).append(x)
    drift = {}
    for g, xs in groups.items():
        xs = torch.stack(xs)
        pair = [float(a @ b) for a, b in itertools.combinations(xs, 2)]
        drift[g] = {'to_subject': round(float((xs @ S.T).mean()), 4),
                    'to_each_other': round(sum(pair) / len(pair), 4), 'n': len(xs)}
    res[f'{key}_drift'] = drift
save_json(res, OUT / 'similarity.json')

# Training loss
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for run in DB_RUNS:
        rows = []
        for ev in sorted((WORK / f'db-{run}' / 'logs').rglob('events.out.tfevents.*')):
            acc = EventAccumulator(str(ev), size_guidance={'scalars': 0})
            acc.Reload()
            for tag in acc.Tags().get('scalars', []):
                if 'loss' in tag:
                    rows += [(e.step, tag, e.value) for e in acc.Scalars(tag)]
        with open(OUT / f'loss-{run}.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['step', 'tag', 'value'])
            w.writerows(sorted(rows))
        print(run, len(rows), 'loss values')
except Exception as e:                                         # noqa: BLE001
    print('loss extraction failed:', e)

# LoRA sizes
from safetensors.torch import load_file                        # noqa: E402

lora = {}
for run in DB_RUNS:
    f = WORK / f'db-{run}' / 'pytorch_lora_weights.safetensors'
    if f.exists():
        sd = load_file(str(f))
        lora[run] = {'bytes': f.stat().st_size, 'params': sum(v.numel() for v in sd.values()),
                     'unet_params': sum(v.numel() for k, v in sd.items() if k.startswith('unet')),
                     'text_params': sum(v.numel() for k, v in sd.items() if k.startswith('text_encoder')),
                     'dtype': str(next(iter(sd.values())).dtype), 'n_tensors': len(sd)}
save_json(lora, OUT / 'lora.json')
print(lora)

"""Gather the small results for the slides and zip them.

    python collect.py

- results/train_loss.csv: the per-step training loss that train_text_to_image.py logged to tensorboard.
- results/nearest.json: for every generated image, the three most similar training prints by CLIP image
  embedding (cosine similarity), to check for memorisation.
- lec18_results.zip: everything in results/ (images, JSON, logs; no checkpoints).
"""
import csv
import json
import shutil

import torch
from PIL import Image

from common import CLIP, DATA, HERE, RESULTS, RUN, save_json


def losses():
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    rows = {}
    for ev in sorted(RUN.glob('logs/**/events.out.tfevents.*')):
        acc = EventAccumulator(str(ev), size_guidance={'scalars': 0})
        acc.Reload()
        if 'train_loss' in acc.Tags().get('scalars', []):
            for e in acc.Scalars('train_loss'):
                rows[e.step] = (e.wall_time, e.value)         # a resumed stage's value replaces an earlier one
    with open(RESULTS / 'train_loss.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'wall_time', 'train_loss'])
        for s in sorted(rows):
            w.writerow([s, rows[s][0], rows[s][1]])
    print('train_loss.csv:', len(rows), 'steps')


@torch.no_grad()
def nearest():
    from transformers import CLIPModel, CLIPProcessor
    model = CLIPModel.from_pretrained(CLIP, torch_dtype=torch.float16).to('cuda').eval()
    proc = CLIPProcessor.from_pretrained(CLIP)

    def embed(paths):
        out = []
        for i in range(0, len(paths), 32):
            ims = [Image.open(p).convert('RGB') for p in paths[i:i + 32]]
            x = proc(images=ims, return_tensors='pt')['pixel_values'].to('cuda', torch.float16)
            e = model.get_image_features(pixel_values=x).float()
            out.append(e / e.norm(dim=-1, keepdim=True))
        return torch.cat(out)

    train = sorted((DATA / 'train').glob('*.jpg'))
    gen = sorted((RESULTS / 'samples').glob('s*/*/*.jpg'))
    et, eg = embed(train), embed(gen)
    sim = eg @ et.T
    top = sim.topk(3, dim=1)
    res = {str(p.relative_to(RESULTS)): [{'file_name': train[j].name, 'cos': round(float(v), 4)}
                                         for v, j in zip(top.values[k].tolist(), top.indices[k].tolist())]
           for k, p in enumerate(gen)}
    # For scale: how similar are distinct training prints to each other?
    tt = et @ et.T
    tt.fill_diagonal_(-1)
    res['_train_to_train_max_cos'] = {'median': float(tt.max(dim=1).values.median()),
                                      'p95': float(tt.max(dim=1).values.quantile(0.95))}
    save_json(res, RESULTS / 'nearest.json')
    print('nearest.json:', len(gen), 'generated images')


def main():
    losses()
    nearest()
    for f in ('probe.json',):
        if not (RESULTS / f).exists():
            print('note: no', f)
    zip_path = shutil.make_archive(str(HERE / 'lec18_results'), 'zip', root_dir=RESULTS.parent, base_dir=RESULTS.name)
    print('wrote', zip_path)


if __name__ == '__main__':
    main()

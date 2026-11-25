"""Build the fine-tuning dataset: CC0 Japanese woodblock prints from the Cleveland Museum of Art, captioned by BLIP.

    python prepare_data.py

1. Lists the museum's open-access Japanese woodblock prints (public API, no key) and downloads each image.
   Very tall or wide prints are skipped, since a 512 x 512 centre crop would lose most of them.
2. Resizes to a 768 px short side and splits off N_HELDOUT prints that are never trained on.
3. Captions every print with BLIP (on the GPU). The captions describe what is shown, not the style.
4. Writes imagefolder datasets (work/data/{train,heldout}/ with metadata.jsonl) and, for the slides,
   results/data/catalogue.json (credits and captions), thumbnails and a few full-size examples.

Resumable: downloaded images and captions are kept and reused.
"""
import json
import random
import re
import shutil
import time
from io import BytesIO

import requests
from PIL import Image

from common import CAPTIONER, DATA, MAX_ASPECT, N_HELDOUT, RESULTS, SHORT_SIDE, save_json

API = 'https://openaccess-api.clevelandart.org/api/artworks/'
RAW = DATA / 'raw'
# The public deck shows training prints and generations: leave out anything erotic or nude. No shunga turned up
# in the titles or descriptions (Nov 2026), but filter anyway; 1943.46 ("Woman Bathing") is partly nude.
EXCLUDE_WORDS = re.compile(r'shunga|erotic|spring picture|nude|naked|bathing', re.I)
EXCLUDE_ACCESSIONS = {'1943.46'}
UA = {'User-Agent': 'teaching-dataset-script/1.0 (python-requests)'}


def list_prints():
    cache = DATA / 'listing.json'
    if cache.exists():
        return json.loads(cache.read_text())
    rows, skip = [], 0
    while True:
        r = requests.get(API, params={'department': 'Japanese Art', 'type': 'Print', 'cc0': 1, 'has_image': 1,
                                      'limit': 500, 'skip': skip}, headers=UA, timeout=60)
        r.raise_for_status()
        d = r.json()
        rows += d['data']
        skip += len(d['data'])
        if not d['data'] or skip >= d['info']['total']:
            break
    keep = []
    for x in rows:
        im = (x.get('images') or {}).get('print') or (x.get('images') or {}).get('web')
        if not im or 'woodblock' not in (x.get('technique') or '').lower():
            continue
        if x['accession_number'] in EXCLUDE_ACCESSIONS or EXCLUDE_WORDS.search(
                ' '.join(str(x.get(k) or '') for k in ('title', 'description'))):
            continue
        w, h = int(im['width']), int(im['height'])
        if max(w, h) / min(w, h) > MAX_ASPECT:
            continue
        keep.append({'id': x['id'], 'accession': x['accession_number'], 'title': x.get('title'),
                     'artist': '; '.join(c.get('description', '') for c in x.get('creators') or []),
                     'date': x.get('creation_date'), 'technique': x.get('technique'),
                     'url': x.get('url'), 'image_url': im['url'], 'width': w, 'height': h})
    keep.sort(key=lambda k: k['accession'])
    DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(keep, indent=1))
    print(f'{len(rows)} Japanese prints listed, {len(keep)} woodblock prints kept')
    return keep


def fname(item):
    return item['accession'].replace('/', '_').replace(' ', '_') + '.jpg'


def download(items):
    RAW.mkdir(parents=True, exist_ok=True)
    ok = []
    for k, it in enumerate(items):
        path = RAW / fname(it)
        if not path.exists():
            for attempt in range(4):
                try:
                    r = requests.get(it['image_url'], headers=UA, timeout=120)
                    r.raise_for_status()
                    im = Image.open(BytesIO(r.content)).convert('RGB')
                    s = SHORT_SIDE / min(im.size)
                    if s < 1:
                        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
                    im.save(path, quality=93)
                    break
                except Exception as e:                   # noqa: BLE001
                    print('  retry', it['accession'], repr(e)[:120])
                    time.sleep(5 * (attempt + 1))
            time.sleep(0.2)                              # be gentle with the museum's server
        if path.exists():
            ok.append(it)
        if k % 50 == 0:
            print(f'downloaded {k}/{len(items)}', flush=True)
    return ok


def caption(items):
    cache = DATA / 'captions.json'
    caps = json.loads(cache.read_text()) if cache.exists() else {}
    todo = [it for it in items if fname(it) not in caps]
    if todo:
        import torch
        from transformers import BlipForConditionalGeneration, BlipProcessor
        proc = BlipProcessor.from_pretrained(CAPTIONER)
        model = BlipForConditionalGeneration.from_pretrained(CAPTIONER, torch_dtype=torch.float16).to('cuda')
        for i in range(0, len(todo), 16):
            batch = todo[i:i + 16]
            ims = [Image.open(RAW / fname(it)).convert('RGB') for it in batch]
            inp = proc(images=ims, return_tensors='pt').to('cuda', torch.float16)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=40, num_beams=3)
            for it, c in zip(batch, proc.batch_decode(out, skip_special_tokens=True)):
                # BLIP-large sometimes starts with the nonsense words "arafed"/"araffe": drop them.
                words = [w for w in c.strip().split() if not w.startswith(('araf', 'arafed'))]
                caps[fname(it)] = ' '.join(words)
            cache.write_text(json.dumps(caps, indent=1))
            print(f'captioned {min(i + 16, len(todo))}/{len(todo)}', flush=True)
    return caps


def centre_square(im, size):
    s = min(im.size)
    l, t = (im.width - s) // 2, (im.height - s) // 2
    return im.crop((l, t, l + s, t + s)).resize((size, size), Image.LANCZOS)


def main():
    items = download(list_prints())
    caps = caption(items)
    rng = random.Random(0)
    order = items[:]
    rng.shuffle(order)
    split = {fname(it): ('heldout' if k < N_HELDOUT else 'train') for k, it in enumerate(order)}

    for part in ('train', 'heldout'):
        d = DATA / part
        d.mkdir(parents=True, exist_ok=True)
        lines = []
        for it in items:
            f = fname(it)
            if split[f] != part:
                continue
            if not (d / f).exists():
                shutil.copyfile(RAW / f, d / f)
            lines.append(json.dumps({'file_name': f, 'text': caps[f]}))
        (d / 'metadata.jsonl').write_text('\n'.join(lines) + '\n')
        print(part, len(lines), 'images')

    out = RESULTS / 'data'
    (out / 'thumbs').mkdir(parents=True, exist_ok=True)
    (out / 'examples').mkdir(parents=True, exist_ok=True)
    cat = []
    for it in items:
        f = fname(it)
        im = Image.open(RAW / f).convert('RGB')
        centre_square(im, 160).save(out / 'thumbs' / f, quality=85)
        cat.append({**{k: it[k] for k in ('id', 'accession', 'title', 'artist', 'date', 'technique', 'url')},
                    'file_name': f, 'caption': caps[f], 'split': split[f],
                    'stored_size': list(im.size)})
    train = [c for c in cat if c['split'] == 'train']
    for c in random.Random(1).sample(train, min(16, len(train))):
        im = Image.open(RAW / c['file_name']).convert('RGB')
        s = 512 / min(im.size)
        im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS).save(
            out / 'examples' / c['file_name'], quality=88)
        c['example'] = True
    save_json({'source': 'Cleveland Museum of Art Open Access (CC0)', 'captioner': CAPTIONER,
               'n_train': len(train), 'n_heldout': len(cat) - len(train), 'items': cat}, out / 'catalogue.json')
    print('wrote', out / 'catalogue.json')


if __name__ == '__main__':
    main()

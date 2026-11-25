"""Build the input images for the lec19-22 GPU bundle (CPU only, seconds).

Writes into lec19_22_gpu/inputs/: 512 x 512 RGB photos and artworks, a simple sketch, and inpainting masks
(white = regenerate, black = keep). The DreamBooth subject photos (inputs/dog/) are downloaded separately
from github.com/google/dreambooth (Unsplash, credit @alvannee).
Run with the genaibook env: python lec19_22_inputs.py
"""
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFilter
from skimage import data

OUT = Path(__file__).resolve().parent / 'lec19_22_gpu' / 'inputs'
OUT.mkdir(parents=True, exist_ok=True)
S = 512


def square(im, cx=0.5, cy=0.5):
    """Crop the largest square around (cx, cy) (fractions of width/height), then resize to S x S."""
    w, h = im.size
    side = min(w, h)
    x0 = int(min(max(cx * w - side / 2, 0), w - side))
    y0 = int(min(max(cy * h - side / 2, 0), h - side))
    return im.crop((x0, y0, x0 + side, y0 + side)).resize((S, S), Image.LANCZOS)


def trim(im, f):
    """Cut a fraction f off every side (a print's paper margin)."""
    w, h = im.size
    return im.crop((int(f * w), int(f * h), int((1 - f) * w), int((1 - f) * h)))


def fetch(url):
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert('RGB')


# scikit-image sample photos (CC0 or public domain), as in the CLIP lecture
square(Image.fromarray(data.chelsea()), cx=0.45).save(OUT / 'cat.png')
square(Image.fromarray(data.coffee()), cx=0.5).save(OUT / 'coffee.png')
square(Image.fromarray(data.rocket()), cx=0.5, cy=0.5).save(OUT / 'rocket.png')
square(Image.fromarray(data.astronaut())).save(OUT / 'astronaut.png')

# Cleveland Museum of Art Open Access (CC0)
square(trim(fetch('https://openaccess-cdn.clevelandart.org/1985.318/1985.318_web.jpg'), 0.06), cy=0.42).save(OUT / 'hiroshige.png')
square(fetch('https://openaccess-cdn.clevelandart.org/1947.209/1947.209_web.jpg')).save(OUT / 'vangogh.png')

# A child-like sketch in flat colours: hills, a cottage, a tree, the sun
im = Image.new('RGB', (S, S), (170, 215, 245))
d = ImageDraw.Draw(im)
d.ellipse([360, 50, 450, 140], fill=(255, 205, 40), outline=(0, 0, 0), width=3)
d.polygon([(0, 330), (120, 280), (260, 300), (400, 260), (512, 290), (512, 512), (0, 512)], fill=(110, 190, 80),
          outline=(0, 0, 0))
d.rectangle([150, 250, 300, 380], fill=(235, 225, 200), outline=(0, 0, 0), width=3)
d.polygon([(135, 255), (225, 180), (315, 255)], fill=(190, 60, 50), outline=(0, 0, 0))
d.rectangle([205, 315, 245, 380], fill=(120, 80, 40), outline=(0, 0, 0), width=3)
d.rectangle([168, 275, 198, 305], fill=(140, 200, 240), outline=(0, 0, 0), width=3)
d.rectangle([255, 275, 285, 305], fill=(140, 200, 240), outline=(0, 0, 0), width=3)
d.rectangle([385, 270, 405, 380], fill=(120, 80, 40), outline=(0, 0, 0), width=3)
d.ellipse([340, 180, 450, 300], fill=(40, 140, 50), outline=(0, 0, 0), width=3)
d.line([(0, 420), (140, 400), (230, 440), (512, 410)], fill=(90, 160, 60), width=3)
im.save(OUT / 'sketch.png')

# Inpainting masks (L mode; white = regenerate). Coffee: the cup, tight and loose.
def mask(draw_fn, blur=0):
    m = Image.new('L', (S, S), 0)
    draw_fn(ImageDraw.Draw(m))
    return m.filter(ImageFilter.GaussianBlur(blur)) if blur else m


mask(lambda d: d.ellipse([80, 5, 450, 390], fill=255)).save(OUT / 'coffee-mask-tight.png')
mask(lambda d: d.ellipse([30, -20, 500, 470], fill=255)).save(OUT / 'coffee-mask-loose.png')
print('wrote', sorted(p.name for p in OUT.iterdir()))

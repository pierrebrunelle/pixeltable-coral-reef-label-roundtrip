"""Render small, deterministic coral-reef survey photos plus the ground-truth labels.

Every photo is a sandy seabed with a few coloured regions drawn at known places: hard coral (pink),
soft coral (orange), algae (green), rubble (grey) and bleached coral (near white). `data/truth.json`
holds the exact boxes, so `pipeline.py verify` can check the whole label round trip.

    python make_photos.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

W, H = 512, 384
COLORS = {
    'hard_coral': (226, 104, 150),
    'soft_coral': (240, 146, 60),
    'algae': (64, 150, 72),
    'rubble': (122, 122, 126),
    'bleached_coral': (238, 238, 232),
}
SITES = [
    ('reef-001', 'North Wall', 6.5),
    ('reef-002', 'North Wall', 9.0),
    ('reef-003', 'Lagoon Patch', 3.0),
    ('reef-004', 'Lagoon Patch', 4.5),
    ('reef-005', 'Outer Slope', 12.0),
    ('reef-006', 'Outer Slope', 15.5),
]
DATA = Path(__file__).parent / 'data'


def render(photo_id: str, seed: int) -> tuple[Image.Image, list[dict]]:
    rnd = random.Random(seed)
    img = Image.new('RGB', (W, H), (196, 180, 132))
    px = img.load()
    for y in range(0, H, 2):  # a little sand texture
        for x in range(0, W, 2):
            d = rnd.randint(-10, 10)
            c = (196 + d, 180 + d, 132 + d)
            for dy in (0, 1):
                for dx in (0, 1):
                    px[x + dx, y + dy] = c
    draw = ImageDraw.Draw(img)
    regions: list[dict] = []
    occupied: list[tuple[int, int, int, int]] = []
    labels = list(COLORS)
    n = rnd.randint(4, 6)
    tries = 0
    while len(regions) < n and tries < 200:
        tries += 1
        label = labels[(len(regions) + seed) % len(labels)]
        # one small region per photo: easy for a model to miss, so the annotator adds it
        small = len(regions) == n - 1
        w = rnd.randint(14, 20) if small else rnd.randint(50, 120)
        h = rnd.randint(14, 20) if small else rnd.randint(40, 90)
        x = rnd.randint(8, W - w - 8)
        y = rnd.randint(8, H - h - 8)
        box = (x - 6, y - 6, x + w + 6, y + h + 6)
        if any(not (box[2] < o[0] or box[0] > o[2] or box[3] < o[1] or box[1] > o[3]) for o in occupied):
            continue
        occupied.append(box)
        color = COLORS[label]
        if label == 'soft_coral' and rnd.random() < 0.5:
            color = (236, 124, 106)  # a pinkish soft coral: the colour model calls it hard coral
        if label in ('hard_coral', 'bleached_coral'):
            draw.ellipse((x, y, x + w - 1, y + h - 1), fill=color)
        else:
            draw.rectangle((x, y, x + w - 1, y + h - 1), fill=color)
        regions.append({'label': label, 'bbox': [x, y, w, h]})
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img, regions


def main() -> None:
    DATA.mkdir(exist_ok=True)
    truth = {}
    for i, (photo_id, site, depth) in enumerate(SITES):
        img, regions = render(photo_id, seed=100 + i)
        img.save(DATA / f'{photo_id}.jpg', quality=92)
        truth[photo_id] = {'site': site, 'depth_m': depth, 'regions': regions}
    (DATA / 'truth.json').write_text(json.dumps(truth, indent=1))
    print(f'wrote {len(truth)} photos + truth.json to {DATA.name}/')


if __name__ == '__main__':
    main()

"""UDFs for the coral-reef label round trip: a small colour-based segmenter (the "model"), Label Studio
task / annotation conversion, label diffs and COCO boxes. Pure Python, no table imports."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import PIL.Image

import pixeltable as pxt

LABELS = ('hard_coral', 'soft_coral', 'algae', 'rubble', 'bleached_coral')
# reference colours of the classes, plus sand (index 5) as background
PALETTE = np.array(
    [(226, 104, 150), (240, 146, 60), (64, 150, 72), (122, 122, 126), (238, 238, 232), (196, 180, 132)],
    dtype=np.float32,
)
SCALE = 4  # segment at 1/4 resolution
MIN_AREA = 600  # regions smaller than this (in full-resolution pixels) are dropped as noise
MODEL_VERSION = 'reef-color-v1'
LS_FROM, LS_TO = 'label', 'image'  # names in the Label Studio labeling config


class Region(TypedDict):
    label: str
    bbox: list[int]  # [x, y, w, h] in pixels
    area: int
    score: float


class Segmentation(TypedDict):
    model: str
    regions: list[Region]


def _components(mask: np.ndarray) -> list[tuple[int, int, int, int, int]]:
    """4-connected components of a boolean mask -> (x0, y0, x1, y1, n_pixels)."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for y0, x0 in zip(*np.nonzero(mask)):
        if seen[y0, x0]:
            continue
        stack = [(y0, x0)]
        seen[y0, x0] = True
        xs, ys, n = [x0, x0], [y0, y0], 0
        while stack:
            y, x = stack.pop()
            n += 1
            xs[0], xs[1] = min(xs[0], x), max(xs[1], x)
            ys[0], ys[1] = min(ys[0], y), max(ys[1], y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        out.append((int(xs[0]), int(ys[0]), int(xs[1]), int(ys[1]), n))
    return out


@pxt.udf
def segment_reef(image: PIL.Image.Image) -> Segmentation:
    """Nearest-colour segmentation into the reef classes, one box per connected region."""
    small = image.convert('RGB').resize((image.width // SCALE, image.height // SCALE), PIL.Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32)
    dist = ((arr[:, :, None, :] - PALETTE[None, None, :, :]) ** 2).sum(-1)
    cls = dist.argmin(-1)
    regions: list[Region] = []
    for i, label in enumerate(LABELS):
        for x0, y0, x1, y1, n in _components(cls == i):
            area = n * SCALE * SCALE
            if area < MIN_AREA:
                continue
            box = [x0 * SCALE, y0 * SCALE, (x1 - x0 + 1) * SCALE, (y1 - y0 + 1) * SCALE]
            fill = n / ((x1 - x0 + 1) * (y1 - y0 + 1))
            regions.append({'label': label, 'bbox': box, 'area': area, 'score': round(min(1.0, fill), 3)})
    regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
    return {'model': MODEL_VERSION, 'regions': regions}


@pxt.udf
def cover_pct(seg: Segmentation, width: int, height: int) -> dict:
    """Percent of the photo covered by each class (box area, rounded to 0.1)."""
    total = float(width * height)
    out = {label: 0.0 for label in LABELS}
    for r in seg['regions']:
        out[r['label']] = out.get(r['label'], 0.0) + r['bbox'][2] * r['bbox'][3] / total * 100
    return {k: round(v, 1) for k, v in out.items()}


def _ls_result(r: dict, width: int, height: int, idx: int) -> dict:
    x, y, w, h = r['bbox']
    item = {
        'id': f'r{idx}',
        'from_name': LS_FROM,
        'to_name': LS_TO,
        'type': 'rectanglelabels',
        'original_width': width,
        'original_height': height,
        'image_rotation': 0,
        'value': {
            'x': round(x / width * 100, 4),
            'y': round(y / height * 100, 4),
            'width': round(w / width * 100, 4),
            'height': round(h / height * 100, 4),
            'rotation': 0,
            'rectanglelabels': [r['label']],
        },
    }
    if 'score' in r:
        item['score'] = r['score']
    return item


@pxt.udf
def ls_task(photo_id: str, file_name: str, site: str, seg: Segmentation, width: int, height: int) -> dict:
    """A Label Studio import task: the image reference plus the model's boxes as pre-annotations."""
    return {
        'data': {'image': f'/data/local-files/?d=reef/{file_name}', 'photo_id': photo_id, 'site': site},
        'predictions': [
            {
                'model_version': seg['model'],
                'result': [_ls_result(r, width, height, i) for i, r in enumerate(seg['regions'])],
            }
        ],
    }


@pxt.udf
def ls_to_segmentation(annotation: dict) -> Segmentation:
    """Turn one Label Studio annotation (`{'result': [...]}`) back into pixel boxes."""
    regions: list[Region] = []
    for item in annotation.get('result', []):
        if item.get('type') != 'rectanglelabels':
            continue
        v, w, h = item['value'], item['original_width'], item['original_height']
        box = [round(v['x'] * w / 100), round(v['y'] * h / 100), round(v['width'] * w / 100), round(v['height'] * h / 100)]
        regions.append({'label': v['rectanglelabels'][0], 'bbox': box, 'area': box[2] * box[3], 'score': 1.0})
    regions.sort(key=lambda r: (r['bbox'][1], r['bbox'][0]))
    return {'model': 'human', 'regions': regions}


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, bx1, by1 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax1, bx1) - max(a[0], b[0]))
    ih = max(0, min(ay1, by1) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union else 0.0


def match_regions(pred: list[dict], ref: list[dict], min_iou: float = 0.5) -> dict:
    """Greedy IoU matching of predicted vs reference boxes."""
    pairs = sorted(
        ((_iou(p['bbox'], r['bbox']), i, j) for i, p in enumerate(pred) for j, r in enumerate(ref)), reverse=True
    )
    used_p, used_r, kept, relabeled, ious = set(), set(), 0, [], []
    for iou, i, j in pairs:
        if iou < min_iou or i in used_p or j in used_r:
            continue
        used_p.add(i)
        used_r.add(j)
        ious.append(iou)
        if pred[i]['label'] == ref[j]['label']:
            kept += 1
        else:
            relabeled.append(f"{pred[i]['label']}->{ref[j]['label']}")
    return {
        'kept': kept,
        'relabeled': relabeled,
        'added': [ref[j]['label'] for j in range(len(ref)) if j not in used_r],
        'removed': [pred[i]['label'] for i in range(len(pred)) if i not in used_p],
        'mean_iou': round(sum(ious) / len(ious), 3) if ious else 0.0,
    }


@pxt.udf
def label_diff(pred: Segmentation, reviewed: Segmentation) -> dict:
    """What the annotator changed: kept / relabeled / added / removed boxes and the mean IoU."""
    return match_regions(pred['regions'], reviewed['regions'])


@pxt.udf
def coco_boxes(seg: Segmentation) -> list[dict]:
    """Boxes in the shape `Query.to_coco_dataset()` expects."""
    return [{'bbox': r['bbox'], 'category': r['label']} for r in seg['regions']]


@pxt.udf
def labels_in_tile(seg: Segmentation, tile_box: list[int]) -> list[str]:
    """Classes whose box overlaps a tile (`tile_box` = [x0, y0, x1, y1])."""
    x0, y0, x1, y1 = tile_box
    out = set()
    for r in seg['regions']:
        x, y, w, h = r['bbox']
        if x < x1 and x + w > x0 and y < y1 and y + h > y0:
            out.add(r['label'])
    return sorted(out)


class SurveyMeta(TypedDict):
    """Dive metadata stored with each photo (a typed `pxt.Json` column)."""

    diver: str
    visibility_m: float

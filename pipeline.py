"""Drive the coral-reef label round trip: ingest -> Label Studio tasks -> review -> re-import -> diff -> export.

    python pipeline.py                               # every step against the local catalog 'reef'
    python pipeline.py ingest export                 # just some steps
    python pipeline.py --target pxt://<org>:<db>     # the same steps against a Pixeltable Cloud database

Steps:
  ingest    insert the photos from data/ (segmentation, cover and Label Studio tasks compute on insert)
  export    write output/ls_tasks.json, a Label Studio import file with the model's boxes as predictions
  annotate  stand-in for the human review: writes output/ls_export.json in Label Studio's export format,
            correcting the predictions against data/truth.json (relabel, add missed regions)
  import    read output/ls_export.json back into Photos.reviewed_ls (one batch_update)
  report    review queue, what annotators changed, model vs reviewed cover
  coco      write the reviewed labels as a COCO dataset (Query.to_coco_dataset)
  lancedb   export the training tiles with their labels to a local LanceDB table
  verify    reviewed boxes == truth for every photo; tiles cover every photo
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import pixeltable as pxt

import app
from udfs import _ls_result, match_regions

ROOT = Path(__file__).parent
DATA, OUT = ROOT / 'data', ROOT / 'output'
STEPS = ['ingest', 'export', 'annotate', 'import', 'report', 'coco', 'lancedb', 'verify']
DIVERS = {'North Wall': 'A. Moreau', 'Lagoon Patch': 'K. Tanaka', 'Outer Slope': 'R. Okafor'}


def truth() -> dict:
    return json.loads((DATA / 'truth.json').read_text())


def ingest() -> None:
    p = app.Photos.table
    rows = [
        {
            'photo_id': pid,
            'site': t['site'],
            'depth_m': t['depth_m'],
            'file_name': f'{pid}.jpg',
            'image': str(DATA / f'{pid}.jpg'),
            'survey': {'diver': DIVERS[t['site']], 'visibility_m': 18.0 - t['depth_m'] / 2},
        }
        for pid, t in truth().items()
    ]
    t0 = time.time()
    status = p.insert(rows, on_error='abort')
    print(f'insert: {status.num_rows} rows incl. views, {status.num_excs} errors in {time.time() - t0:.1f}s')
    print(f'  Photos: {p.count()} rows, Tiles: {app.Tiles.table.count()} rows')


def export() -> None:
    OUT.mkdir(exist_ok=True)
    tasks = [r['task'] for r in app.review_queue().collect()]
    (OUT / 'ls_tasks.json').write_text(json.dumps(tasks, indent=1))
    n = sum(len(t['predictions'][0]['result']) for t in tasks)
    print(f'export: {len(tasks)} Label Studio tasks, {n} pre-annotated boxes -> output/ls_tasks.json')


def annotate() -> None:
    """Simulated annotator: the reviewed result equals the truth boxes, with ids kept where a box matched."""
    tasks = json.loads((OUT / 'ls_tasks.json').read_text())
    tr = truth()
    exported = []
    for i, task in enumerate(tasks):
        pid = task['data']['photo_id']
        w, h = task['predictions'][0]['result'][0]['original_width'], task['predictions'][0]['result'][0]['original_height']
        result = [_ls_result(r, w, h, j) for j, r in enumerate(tr[pid]['regions'])]
        for item in result:
            item.pop('score', None)
        exported.append(
            {
                'id': i + 1,
                'data': task['data'],
                'annotations': [{'id': 100 + i, 'completed_by': 1, 'was_cancelled': False, 'result': result}],
                'predictions': task['predictions'],
            }
        )
    (OUT / 'ls_export.json').write_text(json.dumps(exported, indent=1))
    print(f'annotate: {len(exported)} reviewed tasks -> output/ls_export.json')


def import_reviews() -> None:
    p = app.Photos.table
    exported = json.loads((OUT / 'ls_export.json').read_text())
    rows = [
        {
            'photo_id': t['data']['photo_id'],
            'reviewed_ls': {'result': t['annotations'][0]['result']},
            'annotator': f"user-{t['annotations'][0]['completed_by']}",
        }
        for t in exported
        if t['annotations'] and not t['annotations'][0].get('was_cancelled')
    ]
    t0 = time.time()
    status = p.batch_update(rows)
    print(f'import: {status.num_rows} rows updated in {time.time() - t0:.1f}s ({len(rows)} reviewed tasks)')


def report() -> None:
    print('== review queue:', [r['photo_id'] for r in app.review_queue().collect()])
    print('== label changes')
    for r in app.label_changes().collect():
        c = r['changes']
        print(
            f"  {r['photo_id']}  kept={c['kept']}  relabeled={c['relabeled']}  added={c['added']}"
            f"  removed={c['removed']}  mean_iou={c['mean_iou']}"
        )
    print('== cover (model -> reviewed), % of photo')
    for r in app.cover_by_site().collect():
        pc, rc = r['predicted_cover'], r['reviewed_cover'] or {}
        diffs = {k: f'{pc[k]}->{rc.get(k)}' for k in pc if pc[k] != rc.get(k)}
        print(f"  {r['site']:<13} {r['photo_id']}  {r['depth_m']:>4} m  changed: {diffs or 'none'}")


def coco() -> None:
    from udfs import coco_boxes

    p = app.Photos.table
    q = p.where(p.reviewed_ls != None).select({'image': p.image, 'annotations': coco_boxes(p.reviewed)})  # noqa: E711
    path = q.to_coco_dataset()
    OUT.mkdir(exist_ok=True)
    dest = OUT / 'coco'
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(Path(path).parent, dest)
    d = json.loads((dest / Path(path).name).read_text())
    print(
        f"coco: {len(d['images'])} images, {len(d['annotations'])} boxes, {len(d['categories'])} categories"
        f" -> output/coco/{Path(path).name}"
    )


def lancedb() -> None:
    import lancedb as ldb

    t = app.Tiles.table
    q = t.select(t.photo_id, t.site, t.tile, t.tile_box, t.predicted_labels, t.reviewed_labels)
    uri = OUT / 'lancedb'
    pxt.io.export_lancedb(q, uri, 'reef_tiles', if_exists='overwrite')
    tbl = ldb.connect(str(uri)).open_table('reef_tiles')
    print(f'lancedb: {tbl.count_rows()} tiles -> output/lancedb/reef_tiles, columns {tbl.schema.names}')


def verify() -> None:
    p, ok = app.Photos.table, True
    tr = truth()
    for r in p.select(p.photo_id, p.reviewed).collect():
        ref = [dict(x, area=0, score=1.0) for x in tr[r['photo_id']]['regions']]
        m = match_regions(r['reviewed']['regions'], ref, min_iou=0.9)
        good = m['kept'] == len(ref) and not (m['added'] or m['removed'] or m['relabeled'])
        ok &= good
        print(f"  {r['photo_id']}: {'ok' if good else 'MISMATCH ' + json.dumps(m)}")
    n_tiles = app.Tiles.table.count()
    print(f'verify: {"all photos match truth" if ok else "MISMATCHES"}; {n_tiles} tiles')
    if not ok:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('steps', nargs='*', choices=STEPS + [[]], default=STEPS)
    ap.add_argument('--target', default='reef', help="catalog dir or pxt://<org>:<db> (default 'reef')")
    a = ap.parse_args()
    app.TableModel.bind_all(a.target)
    fns = {'import': import_reviews}
    for step in a.steps or STEPS:
        fns.get(step, globals().get(step))()


if __name__ == '__main__':
    main()

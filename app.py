"""Coral-reef photo labels, round-tripped through Label Studio with Pixeltable.

`Photos` holds one reef survey photo per row with its site, depth and dive metadata. Computed columns
segment it into reef classes (the "model"), measure percent cover and build a Label Studio task with the
model's boxes as pre-annotations. After annotators review the tasks, their Label Studio export is
written back into `reviewed_ls`; computed columns turn it into pixel boxes, diff it against the model
and recompute cover. `Tiles` cuts every photo into overlapping training tiles with the classes they
contain.

Layout: UDFs live in udfs.py; this file holds the table models and the query helpers. pipeline.py drives it.

    python make_photos.py
    pxt schema update app.py reef
    python pipeline.py
"""

from __future__ import annotations

import pixeltable as pxt
import pixeltable.functions as pxtf

from udfs import SurveyMeta, cover_pct, label_diff, labels_in_tile, ls_task, ls_to_segmentation, segment_reef

TableModel = pxt.model_base()


class Photos(TableModel, name='photos'):
    """One reef survey photo, the model's labels and (after review) the annotator's labels."""

    photo_id = pxt.Column(type=pxt.String, primary_key=True)
    site: pxt.String
    depth_m: pxt.Float
    file_name: pxt.String
    image: pxt.Image
    survey: pxt.Json[SurveyMeta]

    width = image.width
    height = image.height
    predicted = segment_reef(image)
    predicted_cover = cover_pct(predicted, width, height)
    task = ls_task(photo_id, file_name, site, predicted, width, height)

    # filled by the Label Studio re-import (pipeline.py import)
    reviewed_ls: pxt.Json | None
    annotator: pxt.String | None
    reviewed = ls_to_segmentation(reviewed_ls)
    changes = label_diff(predicted, reviewed)
    reviewed_cover = cover_pct(reviewed, width, height)

    __indexes__ = [pxt.BtreeIndex(site)]


class Tiles(
    TableModel,
    name='tiles',
    base=Photos,
    iterator=pxtf.image.tile_iterator(Photos.image, tile_size=(256, 192), overlap=(32, 32)),
):
    """Component view: overlapping 256x192 training tiles with the classes each one contains."""

    predicted_labels = labels_in_tile(predicted, tile_box)  # type: ignore[name-defined]  # noqa: F821
    reviewed_labels = labels_in_tile(reviewed, tile_box)  # type: ignore[name-defined]  # noqa: F821


MODELS = (Photos, Tiles)


# Query helpers: plain functions over the bound tables.
def review_queue() -> pxt.Query:
    """Photos still waiting for a human review, worst model confidence first."""
    p = Photos.table
    return p.where(p.reviewed_ls == None).order_by(p.photo_id).select(p.photo_id, p.site, p.task)  # noqa: E711


def label_changes() -> pxt.Query:
    """What annotators changed per photo."""
    p = Photos.table
    return p.where(p.reviewed_ls != None).order_by(p.photo_id).select(p.photo_id, p.annotator, p.changes)  # noqa: E711


def cover_by_site() -> pxt.Query:
    """Model vs reviewed percent cover per photo, grouped by site order."""
    p = Photos.table
    return p.order_by(p.site, p.photo_id).select(p.site, p.photo_id, p.depth_m, p.predicted_cover, p.reviewed_cover)

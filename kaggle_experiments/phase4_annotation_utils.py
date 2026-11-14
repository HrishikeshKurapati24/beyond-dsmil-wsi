"""Compare selected UNI patch squares with CAMELYON16 ASAP XML polygons."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath


TUMOR_GROUPS = {"_0", "_1", "tumor"}
EXCLUSION_GROUPS = {"_2", "exclusion", "non-tumor", "non_tumor"}
GRID = 32


def parse_xml(path: Path) -> dict:
    root = ET.parse(path).getroot()
    result = {"tumor": [], "exclusion": [], "groups": {}}
    for annotation in root.findall(".//Annotation"):
        group = (annotation.get("PartOfGroup") or "").strip().lower()
        result["groups"][group] = result["groups"].get(group, 0) + 1
        if group in TUMOR_GROUPS:
            category = "tumor"
        elif group in EXCLUSION_GROUPS:
            category = "exclusion"
        else:
            raise ValueError(f"Unrecognized CAMELYON16 annotation group: {group!r}")
        coordinates = annotation.findall("./Coordinates/Coordinate")
        points = np.asarray([[float(p.attrib["X"]), float(p.attrib["Y"])]
                             for p in sorted(coordinates, key=lambda p: int(p.attrib["Order"]))],
                            dtype=np.float64)
        if len(points) < 3 or not np.isfinite(points).all():
            raise ValueError(f"Invalid polygon in group {group!r}")
        result[category].append(points)
    if not result["tumor"]:
        raise ValueError("XML has no tumor polygons")
    return result


def membership(points: np.ndarray, polygons: dict) -> np.ndarray:
    inside = np.zeros(len(points), dtype=bool)
    for polygon in polygons["tumor"]:
        inside |= MplPath(np.vstack([polygon, polygon[0]])).contains_points(points)
    for polygon in polygons["exclusion"]:
        inside &= ~MplPath(np.vstack([polygon, polygon[0]])).contains_points(points)
    return inside


def patch_overlap(x: float, y: float, width: int, polygons: dict) -> dict:
    center = np.asarray([[x + width / 2, y + width / 2]], dtype=np.float64)
    offsets = (np.arange(GRID, dtype=np.float64) + .5) * (width / GRID)
    xx, yy = np.meshgrid(x + offsets, y + offsets)
    sampled = np.column_stack([xx.ravel(), yy.ravel()])
    return {"center_in_tumor": bool(membership(center, polygons)[0]),
            "sampled_tumor_fraction": float(membership(sampled, polygons).mean())}


def evaluate_example(item: dict, patch_width: int, polygons: dict | None) -> tuple[list[dict], dict]:
    sid = item["slide_id"]
    records = []
    for anchor_number, (xy, top) in enumerate(zip(item["anchor_coordinates_pixels"],
                                                   item["top_attended_patches"]), 1):
        entries = [("anchor", 0, xy, None)] + [
            ("attended", rank, patch["coordinates_pixels"], patch["attention_weight"])
            for rank, patch in enumerate(top, 1)]
        for role, rank, coords, weight in entries:
            overlap = (patch_overlap(coords[0], coords[1], patch_width, polygons)
                       if polygons is not None else
                       {"center_in_tumor": None, "sampled_tumor_fraction": None})
            records.append({"slide_id": sid, "category": item["category"],
                            "anchor_number": anchor_number, "role": role, "rank": rank,
                            "x_level0": coords[0], "y_level0": coords[1],
                            "attention_weight": weight, **overlap})
    summary = {"slide_id": sid, "category": item["category"],
               "label": item["label"], "prediction": item["prediction"],
               "xml_available": polygons is not None,
               "annotation_complete": polygons is not None and sid != "test_114",
               "tumor_polygon_count": len(polygons["tumor"]) if polygons is not None else None,
               "exclusion_polygon_count": len(polygons["exclusion"]) if polygons is not None else None,
               "overlap_method": "32x32 grid of patch-interior sample points; tumor minus exclusions",
               "positive_patch_rule": "sampled tumor fraction >= 0.10"}
    if polygons is not None:
        anchors = [r for r in records if r["role"] == "anchor"]
        attended = [r for r in records if r["role"] == "attended"]
        summary.update({
            "anchors_with_tumor_center": sum(r["center_in_tumor"] for r in anchors),
            "anchors_with_at_least_10pct_tumor": sum(r["sampled_tumor_fraction"] >= .10 for r in anchors),
            "attended_top8_with_tumor_center": sum(r["center_in_tumor"] for r in attended),
            "attended_top8_with_at_least_10pct_tumor": sum(
                r["sampled_tumor_fraction"] >= .10 for r in attended),
            "attended_top8_total": len(attended),
        })
        for number in range(1, 5):
            group = [r for r in attended if r["anchor_number"] == number]
            weights = np.asarray([r["attention_weight"] for r in group])
            fractions = np.asarray([r["sampled_tumor_fraction"] for r in group])
            summary[f"anchor_{number}_top8_attention_weighted_tumor_fraction"] = (
                float(np.average(fractions, weights=weights)) if weights.sum() > 0 else None)
    return records, summary

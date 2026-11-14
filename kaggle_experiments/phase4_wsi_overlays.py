"""Download selected CAMELYON16 test WSIs and make compressed overlay figures."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import openslide
from phase4_annotation_utils import evaluate_example, parse_xml


BUCKET = "camelyon-dataset"
COLORS = ("#e66101", "#1b9e77", "#7570b3", "#e7298a")


def object_key(s3, slide_id: str) -> tuple[str, int]:
    if not slide_id.startswith("test_") or not slide_id[5:].isdigit():
        raise ValueError(f"Unexpected official test slide ID: {slide_id}")
    candidate = f"CAMELYON16/images/{slide_id}.tif"
    try:
        head = s3.head_object(Bucket=BUCKET, Key=candidate)
        return candidate, int(head["ContentLength"])
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix="CAMELYON16/"):
        keys += [(item["Key"], int(item["Size"])) for item in page.get("Contents", [])
                 if item["Key"].lower().endswith(f"/{slide_id}.tif")]
    if len(keys) != 1:
        raise FileNotFoundError(f"Expected one public TIFF for {slide_id}; found {keys}")
    return keys[0]


def annotation_key(s3, slide_id: str) -> str | None:
    candidate = f"CAMELYON16/annotations/{slide_id}.xml"
    try:
        s3.head_object(Bucket=BUCKET, Key=candidate)
        return candidate
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") not in ("404", "NoSuchKey"):
            raise
    matches = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET,
                                                              Prefix="CAMELYON16/"):
        matches.extend(item["Key"] for item in page.get("Contents", [])
                       if item["Key"].lower().endswith(f"/{slide_id}.xml"))
    if len(matches) > 1:
        raise ValueError(f"Ambiguous XML for {slide_id}: {matches}")
    return matches[0] if matches else None


def draw_overlay(wsi, item: dict, patch_width: int, output: Path,
                 polygons: dict | None) -> None:
    width, height = wsi.dimensions
    image = wsi.get_thumbnail((2600, 2600)).convert("RGB")
    sx, sy = image.width / width, image.height / height
    fig, axes = plt.subplots(2, 3, figsize=(17, 11), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes[:5]:
        ax.imshow(image)
        ax.axis("off")
        if polygons is not None:
            for kind, color, style in (("tumor", "#ffff00", "-"),
                                       ("exclusion", "#00ffff", "--")):
                for polygon in polygons[kind]:
                    loop = list(polygon) + [polygon[0]]
                    ax.plot([xy[0] * sx for xy in loop], [xy[1] * sy for xy in loop],
                            color=color, linestyle=style, linewidth=1.1, alpha=.9)
    axes[0].set_title("Whole-slide image: selected anchors")
    for k, (anchor, patches) in enumerate(zip(item["anchor_coordinates_pixels"],
                                               item["top_attended_patches"])):
        x, y = (anchor[0] + patch_width / 2) * sx, (anchor[1] + patch_width / 2) * sy
        for ax in (axes[0], axes[k + 1]):
            ax.scatter([x], [y], s=260, marker="*", color=COLORS[k],
                       edgecolors="black", linewidths=.8, zorder=5)
            ax.annotate(f"A{k+1}", (x, y), xytext=(5, 5), textcoords="offset points",
                        fontsize=10, weight="bold", color=COLORS[k])
        for patch in patches:
            px, py = patch["coordinates_pixels"]
            center_x = (px + patch_width / 2) * sx
            center_y = (py + patch_width / 2) * sy
            axes[k + 1].scatter([center_x], [center_y],
                                s=35 + 400 * patch["attention_weight"],
                                facecolors=COLORS[k], edgecolors="white", linewidths=.5)
        axes[k + 1].set_title(f"A{k+1}: top {len(patches)} attended patches")
    axes[5].axis("off")
    axes[5].text(.02, .95, "Reading the overlays", va="top", fontsize=14, weight="bold")
    axes[5].text(.02, .82, "Stars: selected anchor patches\n"
                 "Circles: top attention patches for that anchor\n"
                 "Circle size: attention weight\n\n"
                 "Overview is downsampled from the original TIFF.\n"
                 "Coordinates are source level-0 pixels.\n"
                 "Yellow: annotated tumor boundary\n"
                 "Cyan dashed: excluded non-tumor area\n\n"
                 "Patch overlap results are saved separately.", va="top", fontsize=11)
    fig.suptitle(f"{item['slide_id']} | {item['category'].replace('_',' ')} | "
                 f"15-model p(tumor)={item['probability']:.3f}", fontsize=14)
    fig.savefig(output, dpi=150, pil_kwargs={"quality": 83, "optimize": True},
                bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads((args.result_root / "visualization_examples.json").read_text())
    metadata = json.loads((args.reference / "test_coordinate_metadata.json").read_text())
    by_id = {r["slide_id"]: r for r in metadata["slides"]}
    output = args.result_root / "wsi_overlays"
    output.mkdir(exist_ok=True)
    s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))
    generated, all_records, summaries = [], [], []
    for item in data["examples"]:
        sid = item["slide_id"]
        xml_key = annotation_key(s3, sid)
        polygons = None
        if xml_key is not None:
            xml_path = output / f"{sid}.xml"
            xml_path.write_bytes(s3.get_object(Bucket=BUCKET, Key=xml_key)["Body"].read())
            polygons = parse_xml(xml_path)
            print(f"Loaded s3://{BUCKET}/{xml_key}", flush=True)
        else:
            print(f"No XML annotation found for {sid}", flush=True)
        if item["label"] == 1 and polygons is None:
            print(f"WARNING: tumor slide {sid} has no accessible XML; "
                  "overlap metrics will be unavailable", flush=True)
        patch_width = int(by_id[sid]["patch_size_px"])
        records, summary = evaluate_example(item, patch_width, polygons)
        summary["xml_s3_key"] = xml_key
        all_records.extend(records)
        summaries.append(summary)
        key, nbytes = object_key(s3, sid)
        free = shutil.disk_usage("/kaggle/working").free
        if free < nbytes + 2 * 1024**3:
            raise OSError(f"Insufficient Kaggle disk for {sid}: need {nbytes / 1024**3:.1f} GiB "
                          "plus 2 GiB headroom")
        temporary = output / f"{sid}.tif"
        try:
            print(f"Downloading s3://{BUCKET}/{key} ({nbytes / 1024**3:.2f} GiB)", flush=True)
            s3.download_file(BUCKET, key, str(temporary))
            with openslide.OpenSlide(str(temporary)) as wsi:
                expected = tuple(by_id[sid]["level_dim"])
                if tuple(wsi.dimensions) != expected:
                    raise ValueError(f"WSI/UNI coordinate dimensions differ for {sid}: "
                                     f"WSI={wsi.dimensions}, UNI={expected}")
                jpeg = output / f"{item['category']}_{sid}_wsi_overlay.jpg"
                draw_overlay(wsi, item, patch_width, jpeg, polygons)
                generated.append(jpeg)
                print(f"Saved {jpeg} ({jpeg.stat().st_size / 1024**2:.2f} MiB)", flush=True)
        finally:
            temporary.unlink(missing_ok=True)
    (output / "annotation_overlap_summary.json").write_text(json.dumps(
        {"method": "32x32 uniform interior samples per patch; _0/_1/Tumor minus _2 exclusions",
         "caveat": "test_114 has non-exhaustive lesion annotations; no XML is not a negative patch label",
         "slides": summaries}, indent=2) + "\n")
    with (output / "selected_patch_annotation_overlap.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_records[0]))
        writer.writeheader()
        writer.writerows(all_records)
    archive = args.result_root / "three_wsi_overlays.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as bundle:
        for path in generated:
            bundle.write(path, path.name)
        for name in ("annotation_overlap_summary.json", "selected_patch_annotation_overlap.csv"):
            bundle.write(output / name, name)
        for xml_path in output.glob("test_*.xml"):
            bundle.write(xml_path, xml_path.name)
    print(f"Download from Kaggle notebook Output: {archive}", flush=True)


if __name__ == "__main__":
    main()

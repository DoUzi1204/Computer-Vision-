import argparse
import json
from pathlib import Path
from typing import Dict, List, Set


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_image_names(images_dir: Path) -> Set[str]:
    return {
        p.name
        for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def filter_coco(coco_json: Path, images_dir: Path) -> Dict:
    with coco_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    existing_names = list_image_names(images_dir)

    images = [img for img in data.get("images", []) if img.get("file_name") in existing_names]
    kept_image_ids = {img["id"] for img in images}

    annotations = [
        ann for ann in data.get("annotations", []) if ann.get("image_id") in kept_image_ids
    ]

    return {
        "info": data.get("info"),
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": data.get("categories", []),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter COCO annotations by existing images.")
    parser.add_argument("--images-dir", required=True, help="Folder that contains images.")
    parser.add_argument("--coco-json", required=True, help="Path to COCO JSON.")
    parser.add_argument(
        "--output-json",
        default="",
        help="Output JSON path (default: <coco_json>.filtered.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images_dir)
    coco_json = Path(args.coco_json)

    output_json = Path(args.output_json) if args.output_json else coco_json.with_suffix(".filtered.json")

    filtered = filter_coco(coco_json=coco_json, images_dir=images_dir)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False)


if __name__ == "__main__":
    main()

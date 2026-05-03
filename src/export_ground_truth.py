import argparse
from pathlib import Path
from typing import List

from openpyxl import Workbook


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_images(input_dir: Path) -> List[Path]:
    return sorted(
        [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export image names and empty ground truth column."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Folder that contains images (test set).",
    )
    parser.add_argument(
        "--output-xlsx",
        required=True,
        help="Output Excel .xlsx path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_xlsx = Path(args.output_xlsx)

    images = iter_images(input_dir)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "ground_truth"
    sheet.append(["image", "ground_truth"])
    for image_path in images:
        sheet.append([image_path.name, ""])

    workbook.save(output_xlsx)


if __name__ == "__main__":
    main()

import argparse
import os

import cv2
from plate_detection import PlateFinder


def detect_plate_contour_filtering(image, use_adaptive=False, min_area=2000, max_area=26000):
    """
    Detect a license plate region using contour filtering.

    Pipeline
    --------
    Input -> Gray -> Blur -> Threshold -> Contours -> Filter -> Best contour -> BBox -> Crop

    Returns
    -------
    crop : np.ndarray | None
        Cropped plate image or None if not found.
    bbox : tuple[int, int, int, int] | None
        (x, y, w, h) of the best contour candidate.
    debug : dict
        Intermediate images and best score.
    """
    # Keep contour filtering exactly aligned with plate_detection.PlateFinder.
    # use_adaptive is kept for CLI compatibility but not used in this method.
    _ = use_adaptive
    finder = PlateFinder(minPlateArea=min_area, maxPlateArea=max_area)
    possible_plates = finder.find_possible_plates(image, plate_label=None)

    if possible_plates is None:
        return None, None, {"after_preprocess": finder.after_preprocess, "best_score": -1}

    # Choose the largest accepted plate candidate.
    best_idx = max(
        range(len(possible_plates)),
        key=lambda i: possible_plates[i].shape[0] * possible_plates[i].shape[1],
    )
    crop = possible_plates[best_idx].copy()
    x, y = finder.corresponding_area[best_idx]
    h, w = crop.shape[:2]
    bbox = (x, y, w, h)
    score = float(w * h)

    return crop, bbox, {"after_preprocess": finder.after_preprocess, "best_score": score}


def _save_debug_images(debug, out_dir, basename):
    os.makedirs(out_dir, exist_ok=True)
    for key in ["after_preprocess"]:
        if key not in debug:
            continue
        path = os.path.join(out_dir, f"{basename}_{key}.png")
        cv2.imwrite(path, debug[key])


def _process_image(image_path, output_dir, use_adaptive, save_debug, min_area, max_area):
    image = cv2.imread(image_path)
    if image is None:
        return False, None, None

    crop, bbox, debug = detect_plate_contour_filtering(
        image, use_adaptive=use_adaptive, min_area=min_area, max_area=max_area
    )
    base = os.path.splitext(os.path.basename(image_path))[0]
    found = crop is not None and bbox is not None

    if found:
        out_path = os.path.join(output_dir, f"{base}_plate.png")
        cv2.imwrite(out_path, crop)
        print(f"Saved crop: {out_path} bbox={bbox}")
    else:
        print(f"No plate region found for: {image_path}")

    if save_debug:
        _save_debug_images(debug, output_dir, base)

    return found, bbox, debug.get("best_score", -1)


def main():
    parser = argparse.ArgumentParser(description="Contour filtering plate detector")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument(
        "--output",
        default=r"D:\Computer Vision Project\output\crop_contour_base",
        help="Output directory",
    )
    parser.add_argument(
        "--csv",
        default=r"D:\Computer Vision Project\cleaned_report.csv",
        help="Optional CSV report file for reference stats",
    )
    parser.add_argument(
        "--image-dir",
        default=r"D:\Computer Vision Project\data\vietnamese car license plate",
        help="Directory containing test images",
    )
    parser.add_argument(
        "--use-adaptive",
        action="store_true",
        help="Use Adaptive Threshold instead of Otsu",
    )
    parser.add_argument("--save-debug", action="store_true", help="Save intermediate images")
    parser.add_argument("--min-area", type=int, default=2000, help="Minimum plate area")
    parser.add_argument("--max-area", type=int, default=26000, help="Maximum plate area")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    report_path = os.path.join(args.output, "evolution_contour_filtering.txt")

    if args.image:
        found, bbox, score = _process_image(
            args.image, args.output, args.use_adaptive, args.save_debug, args.min_area, args.max_area
        )
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Contour Filtering Detection Report\n")
            f.write(f"Image: {args.image}\n")
            f.write(f"Found: {found}\n")
            f.write(f"BBox: {bbox}\n")
            f.write(f"Score: {score}\n")
        return

    image_files = [
        f
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        raise FileNotFoundError("No image files found in image directory.")

    image_paths = [os.path.join(args.image_dir, f) for f in image_files]
    total_images = len(image_paths)
    detected = 0

    for img_path in image_paths:
        found, _, _ = _process_image(
            img_path, args.output, args.use_adaptive, False, args.min_area, args.max_area
        )
        if found:
            detected += 1

    plate_detection_accuracy = detected / total_images if total_images > 0 else 0

    missing_images = []
    csv_flagged_count = 0
    if args.csv and os.path.exists(args.csv):
        with open(args.csv, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 3:
                    continue
                name, flag, plate = parts[0], parts[1], parts[2]
                if flag == "x" and plate:
                    csv_flagged_count += 1
                    if not os.path.exists(os.path.join(args.image_dir, name)):
                        missing_images.append(name)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Contour Filtering Detection Report\n")
        f.write(f"Images evaluated: {total_images}\n")
        f.write(f"Detected plates: {detected}\n")
        f.write(f"Plate Detection Accuracy (%): {plate_detection_accuracy:.2%}\n")
        f.write(
            "Filter conditions (same as plate_detection.PlateFinder): "
            "preprocess->contours->validateRatio->clean_plate->segment_chars\n"
        )
        f.write(f"CSV flagged samples (flag='x' and plate not empty): {csv_flagged_count}\n")
        f.write(f"Missing CSV flagged images in directory: {len(missing_images)}\n")
        if missing_images:
            f.write("Missing samples (first 20):\n")
            for name in missing_images[:20]:
                f.write(f"- {name}\n")


if __name__ == "__main__":
    main()

import argparse
import os

import cv2
import numpy as np


def detect_plate_edge_morphology(image, use_canny=False):
	"""
	Detect a license plate region using an edge + morphology pipeline.

	Returns
	-------
	crop : np.ndarray | None
		Cropped plate image or None if not found.
	bbox : tuple[int, int, int, int] | None
		(x, y, w, h) of the best component.
	debug : dict
		Intermediate images for inspection.
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (5, 5), 0)

	if use_canny:
		edges = cv2.Canny(blur, 100, 200)
		edge_response = edges.astype(np.float32)
	else:
		sobel = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
		edge_response = np.abs(sobel)
		edge_response = cv2.normalize(edge_response, None, 0, 255, cv2.NORM_MINMAX)
		edges = edge_response.astype(np.uint8)

	_, edge_bin = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

	kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
	closed = cv2.morphologyEx(edge_bin, cv2.MORPH_CLOSE, kernel_close)

	kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

	best_bbox = None
	best_score = -1.0
	for label in range(1, num_labels):
		x, y, w, h, area = stats[label]
		if area <= 0:
			continue
		mask = labels == label
		edge_sum = float(edge_response[mask].sum())
		score = edge_sum * (area ** 0.5)
		if score > best_score:
			best_score = score
			best_bbox = (x, y, w, h)

	crop = None
	if best_bbox is not None:
		x, y, w, h = best_bbox
		crop = image[y : y + h, x : x + w].copy()

	debug = {
		"gray": gray,
		"blur": blur,
		"edges": edges,
		"edge_bin": edge_bin,
		"morph": opened,
		"best_score": best_score,
	}
	return crop, best_bbox, debug


def _save_debug_images(debug, out_dir, basename):
	os.makedirs(out_dir, exist_ok=True)
	for key in ["gray", "blur", "edges", "edge_bin", "morph"]:
		path = os.path.join(out_dir, f"{basename}_{key}.png")
		cv2.imwrite(path, debug[key])


def _is_valid_ratio(width, height, min_ratio, max_ratio):
	if width <= 0 or height <= 0:
		return False
	ratio = float(width) / float(height)
	if ratio < 1.0:
		ratio = 1.0 / ratio
	return min_ratio <= ratio <= max_ratio


def _process_image(
	image_path,
	output_dir,
	use_canny,
	save_debug,
	min_area,
	max_area,
	pre_min_ratio,
	pre_max_ratio,
	final_min_ratio,
	final_max_ratio,
):
	image = cv2.imread(image_path)
	if image is None:
		return False, None, None

	crop, bbox, debug = detect_plate_edge_morphology(image, use_canny=use_canny)
	base = os.path.splitext(os.path.basename(image_path))[0]
	found = False
	if crop is not None and bbox is not None:
		_, _, w, h = bbox
		area = w * h
		# Match PlateFinder-style gating without changing the detection pipeline.
		pre_ratio_ok = _is_valid_ratio(w, h, pre_min_ratio, pre_max_ratio)
		final_ratio_ok = _is_valid_ratio(w, h, final_min_ratio, final_max_ratio)
		if min_area <= area <= max_area and pre_ratio_ok and final_ratio_ok:
			found = True
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
	parser = argparse.ArgumentParser(description="Edge morphology plate detector")
	parser.add_argument("--image", help="Path to input image")
	parser.add_argument(
		"--output",
		default=r"D:\Computer Vision Project\output\crop_edge_morph",
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
	parser.add_argument("--use-canny", action="store_true", help="Use Canny instead of Sobel X")
	parser.add_argument("--save-debug", action="store_true", help="Save intermediate images")
	parser.add_argument(
		"--min-area",
		type=int,
		default=2000,
		help="Minimum valid plate bbox area to count as detected",
	)
	parser.add_argument(
		"--max-area",
		type=int,
		default=26000,
		help="Maximum valid plate bbox area to count as detected",
	)
	parser.add_argument("--pre-min-ratio", type=float, default=1.0, help="Pre ratio min")
	parser.add_argument("--pre-max-ratio", type=float, default=12.0, help="Pre ratio max")
	parser.add_argument("--final-min-ratio", type=float, default=1.2, help="Final ratio min")
	parser.add_argument("--final-max-ratio", type=float, default=10.0, help="Final ratio max")
	args = parser.parse_args()

	os.makedirs(args.output, exist_ok=True)
	report_path = os.path.join(args.output, "evolution_edge_morph.txt")

	if args.image:
		found, bbox, score = _process_image(
			args.image,
			args.output,
			args.use_canny,
			args.save_debug,
			args.min_area,
			args.max_area,
			args.pre_min_ratio,
			args.pre_max_ratio,
			args.final_min_ratio,
			args.final_max_ratio,
		)
		with open(report_path, "w", encoding="utf-8") as f:
			f.write("Edge Morphology Detection Report\n")
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
			img_path,
			args.output,
			args.use_canny,
			False,
			args.min_area,
			args.max_area,
			args.pre_min_ratio,
			args.pre_max_ratio,
			args.final_min_ratio,
			args.final_max_ratio,
		)
		if found:
			detected += 1

	plate_detection_accuracy = detected / total_images if total_images > 0 else 0

	# Optional reference stats from CSV to keep track of dataset completeness.
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
		f.write("Edge Morphology Detection Report\n")
		f.write(f"Images evaluated: {total_images}\n")
		f.write(f"Detected plates: {detected}\n")
		f.write(f"Plate Detection Accuracy (%): {plate_detection_accuracy:.2%}\n")
		f.write(f"Valid detection area range: [{args.min_area}, {args.max_area}]\n")
		f.write(
			f"Valid pre-ratio range: [{args.pre_min_ratio}, {args.pre_max_ratio}] | "
			f"final-ratio range: [{args.final_min_ratio}, {args.final_max_ratio}]\n"
		)
		f.write(f"CSV flagged samples (flag='x' and plate not empty): {csv_flagged_count}\n")
		f.write(f"Missing CSV flagged images in directory: {len(missing_images)}\n")
		if missing_images:
			f.write("Missing samples (first 20):\n")
			for name in missing_images[:20]:
				f.write(f"- {name}\n")


if __name__ == "__main__":
	main()

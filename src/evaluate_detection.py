import os
import cv2
import time
import numpy as np

# Thêm đường dẫn thư mục src vào sys.path để import các module nội bộ
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from yolo_detection import YOLOPlateFinder
from plate_detection import PlateFinder
from Edge_Morphology import detect_plate_edge_morphology

def compute_iou(boxA, boxB):
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def load_yolo_labels(label_path, img_width, img_height):
    """Đọc file label định dạng YOLO (hỗ trợ cả Detection và Segmentation)."""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                # Nếu là segmentation (nhiều hơn 5 phần tử)
                if len(parts) > 5:
                    coords = list(map(float, parts[1:]))
                    x_coords = coords[0::2]
                    y_coords = coords[1::2]
                    x_min = min(x_coords) * img_width
                    x_max = max(x_coords) * img_width
                    y_min = min(y_coords) * img_height
                    y_max = max(y_coords) * img_height
                    x = int(x_min)
                    y = int(y_min)
                    w = int(x_max - x_min)
                    h = int(y_max - y_min)
                    boxes.append([x, y, w, h])
                else:
                    # Detection box
                    x_center, y_center, width, height = map(float, parts[1:5])
                    w = int(width * img_width)
                    h = int(height * img_height)
                    x = int((x_center * img_width) - (w / 2))
                    y = int((y_center * img_height) - (h / 2))
                    boxes.append([x, y, w, h])
    return boxes

def process_single_image(img_path, lbl_path, yolo_finder, plate_finder, metrics, iou_threshold, gt_total_boxes_list):
    img = cv2.imread(img_path)
    if img is None:
        return
    
    h, w = img.shape[:2]
    gt_boxes = load_yolo_labels(lbl_path, w, h)
    gt_total_boxes_list[0] += len(gt_boxes)

    # YOLO
    start_time = time.time()
    yolo_finder.find_possible_plates(img)
    yolo_time = time.time() - start_time
    metrics['YOLOv11']['total_time'] += yolo_time
    
    yolo_preds = yolo_finder.corresponding_area if yolo_finder.corresponding_area else []
    evaluate_single_image_preds(yolo_preds, gt_boxes, metrics['YOLOv11'], iou_threshold)

    # PlateFinder
    start_time = time.time()
    plates = plate_finder.find_possible_plates(img, plate_label=None)
    pf_time = time.time() - start_time
    metrics['PlateFinder_Contour']['total_time'] += pf_time
    
    pf_preds = []
    if plates and plate_finder.corresponding_area:
        for i in range(len(plates)):
            crop = plates[i]
            px, py = plate_finder.corresponding_area[i]
            ph, pw = crop.shape[:2]
            pf_preds.append([px, py, pw, ph])
            
    evaluate_single_image_preds(pf_preds, gt_boxes, metrics['PlateFinder_Contour'], iou_threshold)

    # Edge Morphology
    start_time = time.time()
    _, em_bbox, _ = detect_plate_edge_morphology(img)
    em_time = time.time() - start_time
    metrics['Edge_Morphology']['total_time'] += em_time
    
    em_preds = [em_bbox] if em_bbox is not None else []
    evaluate_single_image_preds(em_preds, gt_boxes, metrics['Edge_Morphology'], iou_threshold)

    # Clean up
    plate_finder.after_preprocess = None
    plate_finder.char_on_plate = []
    yolo_finder.char_on_plate = []
    del img
    del plates
    
def evaluate():
    test_img_dir = r"D:\Computer Vision Project\data\test data\test\images"
    test_lbl_dir = r"D:\Computer Vision Project\data\test data\test\labels"
    yolo_model_path = r"D:\Computer Vision Project\runs\detect\output\yolo11n_plate_detect-3\weights\best.pt"

    print("Khởi tạo mô hình YOLO...")
    yolo_finder = YOLOPlateFinder(model_path=yolo_model_path)
    plate_finder = PlateFinder(minPlateArea=2000, maxPlateArea=26000)

    image_files = [f for f in os.listdir(test_img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    total_images = len(image_files)
    print(f"Tổng số ảnh test: {total_images}")

    metrics = {
        'YOLOv11': {'TP': 0, 'FP': 0, 'FN': 0, 'total_time': 0, 'total_iou': 0, 'pred_count': 0},
        'PlateFinder_Contour': {'TP': 0, 'FP': 0, 'FN': 0, 'total_time': 0, 'total_iou': 0, 'pred_count': 0},
        'Edge_Morphology': {'TP': 0, 'FP': 0, 'FN': 0, 'total_time': 0, 'total_iou': 0, 'pred_count': 0}
    }

    iou_threshold = 0.5
    gt_total_boxes_list = [0]
    
    import torch

    for i, filename in enumerate(image_files):
        img_path = os.path.join(test_img_dir, filename)
        lbl_path = os.path.join(test_lbl_dir, os.path.splitext(filename)[0] + ".txt")
        
        try:
            process_single_image(img_path, lbl_path, yolo_finder, plate_finder, metrics, iou_threshold, gt_total_boxes_list)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {filename}: {e}")

        if (i+1) % 10 == 0:
            print(f"Đã xử lý {i+1}/{total_images} ảnh...")
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    gt_total_boxes = gt_total_boxes_list[0]

    print("-" * 60)
    print(f"{'Phương pháp':<25} | {'IoU':<6} | {'Precision':<9} | {'Recall':<7} | {'F1-Score':<8} | {'mAP@0.5':<7} | {'FPS':<6}")
    print("-" * 60)

    for method, mets in metrics.items():
        tp = mets['TP']
        fp = mets['FP']
        fn = gt_total_boxes - tp  # FN = tổng số GT - số lượng đoán đúng (TP)
        
        # Sửa lỗi logic: tổng FN của phương pháp = tổng GT - TP
        mets['FN'] = fn 
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Với bài toán 1 class, mAP@0.5 tại điểm hoạt động hiện tại chính là tích phân của điểm đó trên PR curve 
        # (hoặc đơn giản là dùng Precision * Recall để xấp xỉ diện tích dưới curve nếu chỉ có 1 điểm).
        # Cách chuẩn xác hơn nếu chỉ có 1 threshold nhị phân: Average Precision = Precision * Recall.
        mAP_05 = precision * recall 

        avg_iou = mets['total_iou'] / tp if tp > 0 else 0
        fps = total_images / mets['total_time'] if mets['total_time'] > 0 else 0

        print(f"{method:<25} | {avg_iou:.4f} | {precision:.4f}    | {recall:.4f}  | {f1:.4f}   | {mAP_05:.4f}  | {fps:.2f}")

def evaluate_single_image_preds(preds, gts, metrics_dict, iou_thresh):
    """So sánh Prediction và Ground truth cho 1 bức ảnh."""
    metrics_dict['pred_count'] += len(preds)
    
    if not gts and not preds:
        return
    if not gts and preds:
        metrics_dict['FP'] += len(preds)
        return
    if not preds and gts:
        # FN sẽ được tính tổng vào cuối
        return

    # Greedy matching giữa preds và gts
    matched_gts = set()
    matched_preds = set()

    for p_idx, p_box in enumerate(preds):
        best_iou = 0
        best_gt_idx = -1
        for g_idx, g_box in enumerate(gts):
            if g_idx in matched_gts:
                continue
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = g_idx
        
        if best_iou >= iou_thresh:
            metrics_dict['TP'] += 1
            metrics_dict['total_iou'] += best_iou
            matched_gts.add(best_gt_idx)
            matched_preds.add(p_idx)
        else:
            metrics_dict['FP'] += 1

if __name__ == "__main__":
    evaluate()

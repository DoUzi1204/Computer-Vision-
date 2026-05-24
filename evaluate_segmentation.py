import os
import cv2
import pandas as pd
from tabulate import tabulate

import sys
sys.path.append(r"D:\Computer Vision Project\src")

from yolo_detection import YOLOPlateFinder
from char_segmentation import segment_chars
from char_segmentation_v2 import segment_chars_vpp

img_dir = r"D:\Computer Vision Project\data\vietnamese car license plate"
csv_path = r"D:\Computer Vision Project\cleaned_report.csv"
yolo_model_path = r"D:\Computer Vision Project\runs\detect\output\yolo11n_plate_detect-3\weights\best.pt"

print("Đang khởi tạo mô hình YOLO để tìm biển số chuẩn...")
yolo_finder = YOLOPlateFinder(model_path=yolo_model_path)

print("Đang đọc dữ liệu Ground Truth...")
report_df = pd.read_csv(csv_path, header=None, names=['image', 'flag', 'plate'])
report_df = report_df[(report_df['flag'] == 'x') & report_df['plate'].notna()]
report_df['plate'] = report_df['plate'].astype(str).str.replace(' ', '').str.replace('-', '').str.replace('.', '')
gt_dict = dict(zip(report_df['image'], report_df['plate']))

contour_correct = 0
vpp_correct = 0
total_evaluated = 0

print("Bắt đầu đánh giá Segmentation Accuracy (chạy qua Contour và VPP)...")

for i, img_name in enumerate(os.listdir(img_dir)):
    if img_name not in gt_dict:
        continue
        
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    gt_str = gt_dict[img_name]
    gt_len = len(gt_str)
    
    # 1. Cắt biển số cực chuẩn bằng YOLO
    plates = yolo_finder.find_possible_plates(img, conf_thresh=0.5)
    if not plates:
        continue 
        
    plate_img = plates[0]
    
    # 2. Phân đoạn ký tự bằng 2 phương pháp
    chars_contour = segment_chars(plate_img)
    chars_vpp = segment_chars_vpp(plate_img)
    
    len_contour = len(chars_contour) if chars_contour is not None else 0
    len_vpp = len(chars_vpp) if chars_vpp is not None else 0
    
    if len_contour == gt_len:
        contour_correct += 1
    if len_vpp == gt_len:
        vpp_correct += 1
        
    total_evaluated += 1

contour_acc = contour_correct / total_evaluated if total_evaluated > 0 else 0
vpp_acc = vpp_correct / total_evaluated if total_evaluated > 0 else 0

results = [
    ["Phương pháp", "Đúng số lượng (Exact Match)", "Segmentation Accuracy (%)"],
    ["Contour Filtering", f"{contour_correct}/{total_evaluated}", f"{contour_acc*100:.2f}%"],
    ["Vertical Projection (VPP)", f"{vpp_correct}/{total_evaluated}", f"{vpp_acc*100:.2f}%"]
]

print("\n" + "="*60)
print("BÁO CÁO ĐÁNH GIÁ SEGMENTATION ACCURACY")
print("Tiêu chí: Số mẩu cắt ra TRÙNG KHỚP với số lượng chữ thực tế")
print("="*60)
print(tabulate(results, headers="firstrow", tablefmt="grid"))
print("="*60)

out_file = r"D:\Computer Vision Project\output\segmentation_evaluation.txt"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    f.write("BÁO CÁO ĐÁNH GIÁ SEGMENTATION ACCURACY\n")
    f.write("Tiêu chí: Số mẩu cắt ra TRÙNG KHỚP với số lượng chữ thực tế (Exact Match)\n\n")
    f.write(tabulate(results, headers="firstrow", tablefmt="grid"))

print(f"Đã lưu báo cáo vào: {out_file}")

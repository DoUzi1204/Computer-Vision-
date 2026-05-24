import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import pandas as pd
import time
from tabulate import tabulate
import matplotlib.pyplot as plt

import sys
sys.path.append(r"D:\Computer Vision Project\src")

from yolo_detection import YOLOPlateFinder
from character_recognition import CharacterRecognizer
from char_segmentation import segment_chars
from ocr_preprocessing import preprocess_for_ocr
from paddleocr import PaddleOCR

img_dir = r"D:\Computer Vision Project\data\vietnamese car license plate"
csv_path = r"D:\Computer Vision Project\cleaned_report.csv"
yolo_model_path = r"D:\Computer Vision Project\runs\detect\output\yolo11n_plate_detect-3\weights\best.pt"
cnn_model_path = r"D:\Computer Vision Project\models\best_char_model.keras"

print("Đang khởi tạo các mô hình (YOLO, CNN, PaddleOCR)...")
yolo_finder = YOLOPlateFinder(model_path=yolo_model_path)
cnn_recognizer = CharacterRecognizer(model_path=cnn_model_path)
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, enable_mkldnn=False)

print("Đang tải Ground Truth...")
report_df = pd.read_csv(csv_path, header=None, names=['image', 'flag', 'plate'])
report_df = report_df[(report_df['flag'] == 'x') & report_df['plate'].notna()]
report_df['plate'] = report_df['plate'].astype(str).str.replace(' ', '').str.replace('-', '').str.replace('.', '')
gt_dict = dict(zip(report_df['image'], report_df['plate']))

total_evaluated = 0
cnn_correct = 0
paddle_correct = 0

cnn_total_time = 0
paddle_total_time = 0

for i, img_name in enumerate(os.listdir(img_dir)):
    if img_name not in gt_dict:
        continue
        
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
        
    gt_str = gt_dict[img_name]
    
    plates = yolo_finder.find_possible_plates(img, conf_thresh=0.5)
    if not plates:
        continue 
    plate_img = plates[0]
    
    # ------------------
    # 1. Đánh giá CNN (Pipeline: YOLO -> Contour -> CNN)
    # ------------------
    start_cnn = time.time()
    cnn_pred = ""
    chars_contour = segment_chars(plate_img)
    if chars_contour:
        cnn_pred = cnn_recognizer.recognize_characters(chars_contour)
    cnn_time = time.time() - start_cnn
    cnn_total_time += cnn_time
    
    if cnn_pred == gt_str:
        cnn_correct += 1

    # ------------------
    # 2. Đánh giá PaddleOCR
    # ------------------
    start_paddle = time.time()
    paddle_pred = ""
    ocr_ready_crop = preprocess_for_ocr(plate_img)
    ocr_results = ocr_reader.ocr(ocr_ready_crop, det=False, cls=False)
    if ocr_results and ocr_results[0]:
        raw_text = "".join([line[0][0] for line in ocr_results[0] if line])
        paddle_pred = "".join(filter(str.isalnum, raw_text)).upper()
    paddle_time = time.time() - start_paddle
    paddle_total_time += paddle_time
    
    if paddle_pred == gt_str:
        paddle_correct += 1
        
    total_evaluated += 1
    
    if total_evaluated % 50 == 0:
        print(f"Đã xử lý {total_evaluated} biển số...")

cnn_acc = cnn_correct / total_evaluated if total_evaluated > 0 else 0
paddle_acc = paddle_correct / total_evaluated if total_evaluated > 0 else 0

cnn_fps = total_evaluated / cnn_total_time if cnn_total_time > 0 else 0
paddle_fps = total_evaluated / paddle_total_time if paddle_total_time > 0 else 0

results = [
    ["Phương pháp", "Đọc chuẩn 100% (Exact Match)", "Plate Recognition Accuracy (%)", "Tốc độ (FPS)"],
    ["CNN Tự Train", f"{cnn_correct}/{total_evaluated}", f"{cnn_acc*100:.2f}%", f"{cnn_fps:.2f} fps"],
    ["PaddleOCR", f"{paddle_correct}/{total_evaluated}", f"{paddle_acc*100:.2f}%", f"{paddle_fps:.2f} fps"]
]

print("\n" + "="*80)
print("BÁO CÁO ĐÁNH GIÁ NHẬN DIỆN BIỂN SỐ (RECOGNITION ACCURACY)")
print("="*80)
print(tabulate(results, headers="firstrow", tablefmt="grid"))
print("="*80)

# Lưu file kết quả
out_file = r"D:\Computer Vision Project\output\recognition_evaluation.txt"
os.makedirs(os.path.dirname(out_file), exist_ok=True)
with open(out_file, "w", encoding="utf-8") as f:
    f.write("BÁO CÁO ĐÁNH GIÁ NHẬN DIỆN BIỂN SỐ (RECOGNITION ACCURACY)\n\n")
    f.write(tabulate(results, headers="firstrow", tablefmt="grid"))

# Vẽ biểu đồ
plt.figure(figsize=(10, 6))
methods = ['CNN Tự Train', 'PaddleOCR']
percentages = [cnn_acc, paddle_acc]
counts = [cnn_correct, paddle_correct]
colors = ['#ff7f0e', '#1f77b4'] 

bars = plt.bar(methods, percentages, color=colors, width=0.5)
plt.title('Plate Recognition Accuracy Comparison', fontsize=14, fontweight='bold')
plt.ylabel('Plate Recognition Accuracy', fontsize=12)
plt.ylim(0, 1.1)
plt.grid(axis='y', linestyle='--', alpha=0.4)

for bar, count, pct in zip(bars, counts, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2, 
             str(count), ha='center', va='center', 
             color='white', fontweight='bold', fontsize=30)
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f"{pct*100:.2f}%", ha='center', fontweight='bold', fontsize=14)

plot_path = r'D:\Computer Vision Project\output\recognition_accuracy_chart.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\nĐã lưu báo cáo: {out_file}")
print(f"Đã lưu biểu đồ: {plot_path}")

import cv2
import os
import numpy as np
import pandas as pd
from yolo_detection import YOLOPlateFinder
from tabulate import tabulate
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
from ocr_preprocessing import preprocess_for_ocr

print("Khởi tạo hệ thống YOLO + PaddleOCR...")
yolo_model_path = r'runs\detect\output\yolo11n_plate_detect-2\weights\best.pt'
if os.path.exists(yolo_model_path):
    findPlate = YOLOPlateFinder(model_path=yolo_model_path)
else:
    print(f"Không tìm thấy model YOLO tại {yolo_model_path}. Xin hãy train mô hình trước.")
    exit(1)

ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

test_dir = r'D:\Computer Vision Project\data\vietnamese car license plate'
output_dir = 'output/crop_images_yolo/'
output_csv = 'output/plates_yolo.csv'
results_csv = 'output/recognition_results_yolo.csv'
report_txt = 'output/evaluation_report_yolo.txt'

os.makedirs(output_dir, exist_ok=True)

report_df = pd.read_csv('./cleaned_report.csv', header=None, names=['image', 'flag', 'plate'])
report_df = report_df[(report_df['flag'] == 'x') & report_df['plate'].notna()]
report_df['plate'] = report_df['plate'].str.replace(' ', '').str.replace('-', '').str.replace('.', '')
image_to_plate = dict(zip(report_df['image'], report_df['plate']))

total_images = 0
plates_detected = 0
total_plates, correct_plates = 0, 0
total_chars, correct_chars = 0, 0
results = []

image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
total_images = len(image_files)

for image_file in image_files:
    print(f"Đang xử lý ảnh: {image_file}")
    img_path = os.path.join(test_dir, image_file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    ground_truth = image_to_plate.get(image_file, '')
    possible_plates = findPlate.find_possible_plates(img)

    if possible_plates is not None:
        plates_detected += 1
        for i, plate in enumerate(possible_plates):
            plate_img = plate.copy()
            cv2.imwrite(os.path.join(output_dir, f"yolo_plate_{image_file}_{i}.jpg"), plate_img)

            # Tiền xử lý 4 bước
            ocr_ready_crop = preprocess_for_ocr(plate_img)

            # Sử dụng PaddleOCR trên ảnh crop đã được tiền xử lý
            ocr_results = ocr_reader.ocr(ocr_ready_crop, det=False, cls=False)
            
            # Extract text from PaddleOCR output
            raw_text = "".join([line[0][0] for line in ocr_results[0] if line]) if ocr_results and ocr_results[0] else ""
            recognized_plate = "".join(filter(str.isalnum, raw_text)).upper()
            
            if recognized_plate:
                total_plates += 1
                results.append([image_file, recognized_plate, ground_truth])
                if recognized_plate == ground_truth:
                    correct_plates += 1
                min_len = min(len(recognized_plate), len(ground_truth))
                for pred, gt in zip(recognized_plate[:min_len], ground_truth[:min_len]):
                    total_chars += 1
                    if pred == gt:
                        correct_chars += 1
                total_chars += abs(len(recognized_plate) - len(ground_truth))
            else:
                results.append([image_file, 'Không nhận diện', ground_truth])
    else:
        results.append([image_file, 'Không nhận diện', ground_truth])

plate_detection_accuracy = plates_detected / total_images if total_images > 0 else 0
plate_recognition_accuracy = correct_plates / total_plates if total_plates > 0 else 0
char_accuracy = correct_chars / total_chars if total_chars > 0 else 0
overall_accuracy = correct_plates / total_images if total_images > 0 else 0

print("\nKết quả cuối cùng (YOLO + EasyOCR)")
print(tabulate(results, headers=["Tên ảnh", "Biển số nhận diện", "Ground Truth"], tablefmt="grid"))

report_summary = [
    ["Tổng số ảnh xử lý", total_images],
    ["Số ảnh phát hiện vùng biển số (YOLO)", plates_detected],
    ["Plate Detection Accuracy (%)", f"{plate_detection_accuracy:.2%}"],
    ["Số ảnh nhận diện biển số", total_plates],
    ["Số biển số nhận diện đúng", correct_plates],
    ["Plate Recognition Accuracy (%)", f"{plate_recognition_accuracy:.2%}"],
    ["Tổng số ký tự", total_chars],
    ["Số ký tự nhận diện đúng", correct_chars],
    ["Character Recognition Accuracy (%)", f"{char_accuracy:.2%}"],
    ["Overall Accuracy (%)", f"{overall_accuracy:.2%}"]
]
print("\n=== Báo cáo đánh giá ===")
print(tabulate(report_summary, headers=["Metric", "Value"], tablefmt="grid"))

df_results = pd.DataFrame(results, columns=['Tên ảnh', 'Biển số nhận diện', 'Ground Truth'])
df_results.to_csv(output_csv, index=False, encoding='utf-8-sig')
df_results.to_csv(results_csv, index=False, encoding='utf-8-sig')
print(f"\nResults saved to: {output_csv}")
print(f"Detailed recognition results saved to: {results_csv}")

with open(report_txt, 'w', encoding='utf-8') as f:
    f.write("=== Báo cáo đánh giá (YOLO + EasyOCR) ===\n")
    f.write(tabulate(report_summary, headers=["Metric", "Value"], tablefmt="plain"))
    f.write(f"\n\nResults saved to: {output_csv}")
    f.write(f"\nDetailed recognition results saved to: {results_csv}")

plt.figure(figsize=(12, 6))
metrics = ['Plate Detection', 'Plate Recognition', 'Character Recognition', 'Overall Accuracy']
values = [plate_detection_accuracy, plate_recognition_accuracy, char_accuracy, overall_accuracy]
bars = plt.bar(metrics, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.title('YOLO Recognition Accuracies')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 1.1)
plt.xticks(rotation=15)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.02, f'{height:.2%}', ha='center')
plt.savefig(os.path.join(output_dir, 'yolo_accuracies_bar_plot.png'))
plt.close()

plt.figure(figsize=(8, 8))
angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
values = [plate_detection_accuracy, plate_recognition_accuracy, char_accuracy, overall_accuracy]
values += values[:1]
angles += angles[:1]
ax = plt.subplot(111, polar=True)
ax.fill(angles, values, color='#1f77b4', alpha=0.25)
ax.plot(angles, values, color='#1f77b4', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics)
for i, v in enumerate(values[:-1]):
    angle = angles[i]
    plt.text(angle, v + 0.05, f'{v:.2%}', ha='center', va='center')
plt.title('YOLO Recognition Accuracies (Radar)')
plt.savefig(os.path.join(output_dir, 'yolo_accuracies_radar_plot.png'))
plt.close()

plt.figure(figsize=(8, 6))
labels = ['Detected Plates', 'Undetected Plates']
sizes = [plates_detected, total_images - plates_detected]
colors = ['#2ca02c', '#d62728']
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
plt.title('YOLO Plate Detection Success Rate')
plt.savefig(os.path.join(output_dir, 'yolo_plate_detection_donut_chart.png'))
plt.close()

print(f"Charts saved to: {output_dir}")
print(f"\nProcessed {len(results)} images")

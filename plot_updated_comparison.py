import matplotlib.pyplot as plt
import os

file_path = r'D:\Computer Vision Project\output\evaluation_results_full.txt'

methods = []
recall_scores = []

# Đọc file (PowerShell '>' mặc định là utf-16)
with open(file_path, 'r', encoding='utf-16') as f:
    lines = f.readlines()

for line in lines:
    if 'YOLOv11' in line or 'PlateFinder_Contour' in line or 'Edge_Morphology' in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 6:
            method_name = parts[0].replace('PlateFinder_Contour', 'Contour Filtering').replace('Edge_Morphology', 'Edge Morphology')
            # Trong bảng, Recall là cột số 3 (Index: Phương pháp=0, IoU=1, Precision=2, Recall=3)
            recall = float(parts[3]) * 100
            
            if method_name in methods:
                idx = methods.index(method_name)
                recall_scores[idx] = recall
            else:
                methods.append(method_name)
                recall_scores.append(recall)

plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#2ca02c', '#d62728'] # Blue, Green, Red
bars = plt.bar(methods, recall_scores, color=colors[:len(methods)])

plt.title('Plate Detection Accuracy (Recall) on 295 images', fontsize=15, fontweight='bold')
plt.ylabel('Accuracy / Recall (%)', fontsize=12)
plt.ylim(0, 115)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 1.5, f'{height:.2f}%', ha='center', fontweight='bold', fontsize=13)

output_path = r'D:\Computer Vision Project\output\plate_detection_accuracy_comparison_full.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart successfully saved to: {output_path}")

# Print the metrics so the LLM can see them
for m, r in zip(methods, recall_scores):
    print(f"{m}: {r}%")

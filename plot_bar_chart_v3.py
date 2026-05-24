import matplotlib.pyplot as plt
import os

methods = ['YOLOv11', 'Contour Filtering', 'Edge Morphology']
percentages = [0.9735, 0.8907, 0.4967]
counts = [287, 263, 146]

plt.figure(figsize=(10, 6))
# Dùng màu xanh dương (YOLO), xanh lá (Contour), đỏ (Edge) giống chuẩn biểu đồ trước
colors = ['#1f77b4', '#2ca02c', '#d62728']

bars = plt.bar(methods, percentages, color=colors)
plt.title('Plate Detection Accuracy Comparison', fontsize=14)
plt.ylabel('Plate Detection Accuracy', fontsize=12)
plt.ylim(0, 1.1)

# Lưới kẻ mờ
plt.grid(axis='y', linestyle='--', alpha=0.4)

# Thêm số lượng ảnh và phần trăm
for bar, count, pct in zip(bars, counts, percentages):
    # Số lượng đếm khổng lồ màu trắng ở giữa thân cột
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2, 
             str(count), ha='center', va='center', 
             color='white', fontweight='bold', fontsize=30)
             
    # Tỷ lệ % ở trên đỉnh cột
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f"{pct*100:.2f}%", ha='center', fontweight='bold', fontsize=12)

output_path = r'D:\Computer Vision Project\output\plate_detection_accuracy_comparison_3bars.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Chart saved to {output_path}")

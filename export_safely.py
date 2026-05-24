import os
import subprocess
import sys

# Tắt tính năng PIR mới của Paddle 3.x để tương thích với mô hình cũ
os.environ["FLAGS_enable_pir_api"] = "0"

cmd = [
    sys.executable,
    "tools/export_model.py",
    "-c", r"d:\Computer Vision Project\vi_plate_rec.yml",
    "-o", 
    r"Global.pretrained_model=d:\Computer Vision Project\models\best_accuracy\best_accuracy",
    r"Global.save_inference_dir=d:\Computer Vision Project\models\inference\rec_vi_plate"
]

print("Exporting model in safe mode...")
subprocess.run(cmd, cwd=r"d:\Computer Vision Project\PaddleOCR")
print("Export complete!")

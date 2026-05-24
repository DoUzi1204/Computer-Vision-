import zipfile
import os

zip_file_path = r"d:\Computer Vision Project\kaggle_dataset.zip"

def zip_dir(dirpath, zf):
    base_dir = os.path.dirname(dirpath)
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            file_path = os.path.join(root, file)
            # Create a relative path
            rel_path = os.path.relpath(file_path, base_dir)
            # Force forward slashes
            arc_name = rel_path.replace('\\', '/')
            zf.write(file_path, arc_name)

with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write(r"d:\Computer Vision Project\data\License Plate OCR\vi_plate_dict.txt", "vi_plate_dict.txt")
    zf.write(r"d:\Computer Vision Project\data\License Plate OCR\kaggle_plate_rec.yml", "kaggle_plate_rec.yml")
    zip_dir(r"d:\Computer Vision Project\data\License Plate OCR\lp_ocr_dataset_vi", zf)

print("Zip created successfully with forward slashes!")

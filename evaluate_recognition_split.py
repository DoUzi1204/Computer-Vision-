import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["FLAGS_use_mkldnn"] = "0"
import sys
import cv2
import pandas as pd
import time
import argparse
from tabulate import tabulate
import matplotlib.pyplot as plt

sys.path.append(r"D:\Computer Vision Project\src")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, required=True, choices=["cnn", "paddle"])
    args = parser.parse_args()

    img_dir = r"D:\Computer Vision Project\data\vietnamese car license plate"
    csv_path = r"D:\Computer Vision Project\cleaned_report.csv"
    yolo_model_path = r"D:\Computer Vision Project\runs\detect\output\yolo11n_plate_detect-3\weights\best.pt"

    from yolo_detection import YOLOPlateFinder
    yolo_finder = YOLOPlateFinder(model_path=yolo_model_path)

    report_df = pd.read_csv(csv_path, header=None, names=['image', 'flag', 'plate'])
    report_df = report_df[(report_df['flag'] == 'x') & report_df['plate'].notna()]
    report_df['plate'] = report_df['plate'].astype(str).str.replace(' ', '').str.replace('-', '').str.replace('.', '')
    gt_dict = dict(zip(report_df['image'], report_df['plate']))

    total_evaluated = 0
    correct = 0
    total_time = 0

    if args.method == "cnn":
        from character_recognition import CharacterRecognizer
        from char_segmentation import segment_chars
        cnn_model_path = r"D:\Computer Vision Project\models\best_char_model.keras"
        recognizer = CharacterRecognizer(model_path=cnn_model_path)
    else:
        from ocr_preprocessing import preprocess_for_ocr
        from paddleocr import PaddleOCR
        ocr_reader = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)

    for img_name in os.listdir(img_dir):
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
        
        start = time.time()
        pred = ""
        
        if args.method == "cnn":
            chars_contour = segment_chars(plate_img)
            if chars_contour:
                pred = recognizer.recognize_characters(chars_contour)
        else:
            ocr_ready_crop = preprocess_for_ocr(plate_img)
            ocr_results = ocr_reader.ocr(ocr_ready_crop, det=False, cls=False)
            if ocr_results and ocr_results[0]:
                raw_text = "".join([line[0][0] for line in ocr_results[0] if line])
                pred = "".join(filter(str.isalnum, raw_text)).upper()
                
        time_taken = time.time() - start
        total_time += time_taken
        
        if pred == gt_str:
            correct += 1
            
        total_evaluated += 1

    with open(f"result_{args.method}.txt", "w") as f:
        f.write(f"{correct},{total_evaluated},{total_time}\n")

if __name__ == "__main__":
    main()

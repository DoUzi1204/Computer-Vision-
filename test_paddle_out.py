from paddleocr import PaddleOCR
import cv2

ocr = PaddleOCR(
    use_gpu=False,
    rec_model_dir=r'd:\Computer Vision Project\models\inference\rec_vi_plate',
    rec_char_dict_path=r'd:\Computer Vision Project\data\License Plate OCR\vi_plate_dict.txt',
    rec_algorithm='SVTR_LCNet',
    rec_image_shape='3, 48, 320',
    use_space_char=False,
    det=False,
    cls=False,
    show_log=False
)

img = cv2.imread(r'd:\Computer Vision Project\data\License Plate OCR\lp_ocr_dataset_vi\imgs\train\car_1.jpg')
res = ocr.ocr(img, det=False, cls=False)
print("PADDLE_RES:", res)

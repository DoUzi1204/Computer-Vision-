"""
plate_detection.py
------------------
Chứa class PlateFinder – phát hiện và xác nhận vùng biển số xe
từ ảnh gốc (ảnh toàn cảnh).

Phụ thuộc:
    char_segmentation.segment_chars  – phân đoạn ký tự trên biển số
"""

import cv2
import numpy as np

from char_segmentation import segment_chars


class PlateFinder:
    """
    Phát hiện biển số xe trong ảnh đầu vào.

    Parameters
    ----------
    minPlateArea : int  – diện tích tối thiểu của vùng biển số (pixel²)
    maxPlateArea : int  – diện tích tối đa của vùng biển số (pixel²)
    """

    def __init__(self, minPlateArea: int, maxPlateArea: int):
        self.min_area = minPlateArea
        self.max_area = maxPlateArea
        self.element_structure = cv2.getStructuringElement(
            shape=cv2.MORPH_RECT, ksize=(18, 3)
        )

    # ── Tiền xử lý ────────────────────────────────────────────────────────────

    def preprocess(self, input_img: np.ndarray) -> np.ndarray:
        """
        Tiền xử lý ảnh để tìm vùng ứng viên biển số.
        Quy trình: Grayscale → Blur → Sobel → Otsu → Morphology
        """
        gray   = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        sobel  = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
        _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Đóng: nối các nét chữ
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)

        # Mở: loại bỏ nhiễu nhỏ
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)

        return opened

    def extract_contours(self, after_preprocess: np.ndarray):
        """Trích xuất các contour ngoài cùng từ ảnh đã tiền xử lý."""
        contours, _ = cv2.findContours(
            after_preprocess,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        return contours

    # ── Kiểm tra & làm sạch biển số ──────────────────────────────────────────

    def clean_plate(self, plate: np.ndarray):
        """
        Làm sạch ảnh biển số và kiểm tra tỉ lệ hình học.

        Returns
        -------
        (plate, found, coordinates)
        """
        gray   = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            areas     = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            max_cnt   = contours[max_index]
            max_cntArea = areas[max_index]
            x, y, w, h = cv2.boundingRect(max_cnt)
            if not self.ratioCheck(max_cntArea, plate.shape[1], plate.shape[0]):
                return plate, False, None
            return plate, True, [x, y, w, h]
        return plate, False, None

    def rotate_and_crop(self, image: np.ndarray, contour):
        """
        Deskew a contour region using minAreaRect angle, then crop it.
        """
        rect = cv2.minAreaRect(contour)
        center, (w, h), angle = rect
        if w == 0 or h == 0:
            return None
        if angle < -45:
            angle += 90

        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

        box = cv2.boxPoints(rect)
        box = np.intp(box)
        ones = np.ones((box.shape[0], 1), dtype=np.float32)
        box_h = np.hstack([box.astype(np.float32), ones])
        rotated_box = (rot_mat @ box_h.T).T
        x, y, w2, h2 = cv2.boundingRect(np.intp(rotated_box))
        if w2 <= 0 or h2 <= 0:
            return None
        return rotated[y : y + h2, x : x + w2]

    def check_plate(self, input_img: np.ndarray, contour, plate_label):
        """
        Xác nhận một contour có phải biển số không; nếu có,
        trả về ảnh biển số và danh sách ký tự.
        """
        min_rect = cv2.minAreaRect(contour)
        if self.validateRatio(min_rect):
            roi = self.rotate_and_crop(input_img, contour)
            if roi is None or roi.size == 0:
                return None, None, None

            x, y, w, h = cv2.boundingRect(contour)
            clean_img, plateFound, coordinates = self.clean_plate(roi)
            if plateFound:
                characters = self.find_characters_on_plate(clean_img)
                if characters:
                    x1, y1, w1, h1 = coordinates
                    abs_coords = (x1 + x, y1 + y)
                    return clean_img, characters, abs_coords
        return None, None, None

    # ── Tìm biển số trong ảnh ─────────────────────────────────────────────────

    def find_possible_plates(self, input_img: np.ndarray, plate_label):
        """
        Tìm tất cả biển số có thể có trong ảnh đầu vào.

        Returns
        -------
        list[np.ndarray] | None  – danh sách ảnh biển số tìm được
        """
        plates = []
        self.char_on_plate      = []
        self.corresponding_area = []
        self.after_preprocess   = self.preprocess(input_img)

        for contour in self.extract_contours(self.after_preprocess):
            plate, chars, coords = self.check_plate(input_img, contour, plate_label)
            if plate is not None:
                plates.append(plate)
                self.char_on_plate.append(chars)
                self.corresponding_area.append(coords)

        return plates if plates else None

    def find_characters_on_plate(self, plate: np.ndarray):
        """Gọi segment_chars để tách ký tự trên biển số."""
        return segment_chars(plate, fixed_width=400)

    # ── Kiểm tra tỉ lệ ───────────────────────────────────────────────────────

    def ratioCheck(self, area: float, width: float, height: float) -> bool:
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        return (
            self.min_area <= area <= self.max_area
            and 1.2 <= ratio <= 10.0
        )

    def preRatioCheck(self, area: float, width: float, height: float) -> bool:
        ratio = float(width) / float(height)
        if ratio < 1:
            ratio = 1 / ratio
        return (
            self.min_area <= area <= self.max_area
            and 1.0 <= ratio <= 12.0
        )

    def validateRatio(self, rect) -> bool:
        """Kiểm tra góc nghiêng và tỉ lệ của hình chữ nhật bao biển số."""
        (x, y), (width, height), rect_angle = rect
        angle = -rect_angle if width > height else 90 + rect_angle
        return (
            abs(angle) <= 40
            and width > 0
            and height > 0
            and self.preRatioCheck(width * height, width, height)
        )

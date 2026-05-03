"""
char_segmentation.py
--------------------
Chứa các hàm hỗ trợ phân đoạn (segment) từng ký tự trên ảnh biển số
đã được cắt ra từ bước phát hiện biển số (PlateFinder).
"""

import cv2
import numpy as np
from skimage import measure
import imutils


def sort_cont(character_contours):
    """Sắp xếp contour ký tự từ trái sang phải theo tọa độ x."""
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours]
    (character_contours, boundingBoxes) = zip(
        *sorted(
            zip(character_contours, boundingBoxes),
            key=lambda b: b[1][i],
            reverse=False,
        )
    )
    return character_contours


def segment_chars(plate_img, fixed_width=400):
    """
    Phân đoạn các ký tự trên ảnh biển số.

    Parameters
    ----------
    plate_img   : np.ndarray  – ảnh biển số (BGR)
    fixed_width : int         – chiều rộng resize trước khi segment

    Returns
    -------
    list[np.ndarray] | None
        Danh sách ảnh từng ký tự (5–9 ký tự), hoặc None nếu không hợp lệ.
    """
    # Chuyển sang grayscale và giảm nhiễu
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2,
    )
    thresh = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Resize về fixed_width
    plate_img = imutils.resize(plate_img, width=fixed_width)
    thresh = imutils.resize(thresh, width=fixed_width)
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    # Labeling & lọc ứng viên ký tự
    labels = measure.label(thresh, background=0)
    charCandidates = np.zeros(thresh.shape, dtype="uint8")
    characters = []

    for label in np.unique(labels):
        if label == 0:
            continue
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        labelMask[labels == label] = 255
        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[1] if imutils.is_cv3() else cnts[0]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
            aspectRatio  = boxW / float(boxH)
            solidity     = cv2.contourArea(c) / float(boxW * boxH)
            heightRatio  = boxH / float(plate_img.shape[0])
            keepAspectRatio = aspectRatio < 2.0
            keepSolidity    = solidity > 0.05
            keepHeight      = 0.3 < heightRatio < 0.98
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 3:
                hull = cv2.convexHull(c)
                cv2.drawContours(charCandidates, [hull], -1, 255, -1)

    contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        contours = sort_cont(contours)
        addPixel = 4
        prev_x = -1
        for c in contours:
            (x, y, w, h) = cv2.boundingRect(c)
            y = max(0, y - addPixel)
            x = max(0, x - addPixel)
            if prev_x != -1 and x - prev_x > w * 2:
                continue
            char_crop = bgr_thresh[y : y + h + addPixel * 2, x : x + w + addPixel * 2]
            characters.append(char_crop)
            prev_x = x + w
        return characters if 5 <= len(characters) <= 9 else None
    return None

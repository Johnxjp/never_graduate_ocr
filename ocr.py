"""
Guide: https://nanonets.com/blog/ocr-with-tesseract/
"""

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np
import pytesseract

Image = np.ndarray
TESSERACT_CONFIG = (
    r"--oem 3 --psm 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


@dataclass
class BoundingBox:
    left: int
    bottom: int
    right: int
    top: int

    def __repr__(self) -> str:
        return f"l={self.left} b={self.bottom} r={self.right} t={self.top}"


@dataclass
class Detections:
    char: str
    box: BoundingBox

    def __repr__(self) -> str:
        return self.char + " " + str(self.box)


def load_img() -> Image:
    return cv2.imread("image4.jpg")


def to_grayscale(image: Image) -> Image:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def sharpen(image: Image) -> Image:
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


def remove_noise(image: Image) -> Image:
    return cv2.medianBlur(image, 5)


def thresholding(image: Image) -> Image:
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def opening(image: Image) -> Image:
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)


def extract_bounding_boxes(image: Image) -> Tuple[Sequence[str], Sequence[BoundingBox]]:
    bounding_boxes_str = pytesseract.image_to_boxes(image, config=TESSERACT_CONFIG)
    chars, boxes = [], []
    for item in bounding_boxes_str.split("\n"):
        char, left, bottom, right, top, _ = item.split(" ")
        chars.append(char)
        boxes.append(BoundingBox(int(left), int(bottom), int(right), int(top)))

    return chars, boxes


def draw_bounding_boxes(image: Image, bounding_boxes: Sequence[BoundingBox]) -> Image:
    height = image.shape[0]
    for box in bounding_boxes:
        image = cv2.rectangle(
            image,
            (box.left, height - box.top),
            (box.right, height - box.bottom),
            (0, 0, 255),
            thickness=2,
        )

    return image


def at_same_level(b1: BoundingBox, b2: BoundingBox) -> bool:
    cond = lambda x, y: (x.top <= y.top and x.top >= y.bottom)
    return cond(b1, b2) or cond(b2, b1)


def sort_top_left(items: Sequence[Detections]) -> Sequence[Detections]:
    # Sort top down first
    sorted_items = sorted(items, key=lambda x: x.box.top, reverse=True)
    current_item, current_box = sorted_items[0], sorted_items[0].box
    final_order = [[current_item]]
    for detection in sorted_items[1:]:
        c = detection.char
        box = detection.box
        if at_same_level(current_box, box):
            final_order[-1].append(Detections(c, box))
        else:
            final_order.append([Detections(c, box)])

        current_box = box
    return [
        item
        for items in final_order
        for item in sorted(items, key=lambda x: x.box.left)
    ]


def is_false_positive(image: Image, item: Detections) -> bool:
    height = image.shape[0]
    box = item.box
    # extract part
    h = box.top - box.bottom
    w = box.right - box.left
    top = height - box.top  # Reverse
    new_image = image[top : top + h, box.left : box.left + w]

    white = 255
    return np.size(new_image) == np.sum(new_image == white)


def remove_false_positives(
    image: Image, items: Sequence[Detections]
) -> Sequence[Detections]:
    return [item for item in items if not is_false_positive(image, item)]


def overlaps(b1: BoundingBox, b2: BoundingBox) -> bool:
    if (b1.right < b2.left) or (b1.left > b2.right):
        return False
    if (b1.bottom > b2.top) or (b1.top < b2.bottom):
        return False
    return True


def filter_overlapping_boxes(detections: Sequence[Detections]) -> Sequence[Detections]:
    filtered_boxes = []
    for prev_item, curr_item in zip(detections[:-1], detections[1:]):
        if not overlaps(curr_item.box, prev_item.box):
            filtered_boxes.append(prev_item)

    filtered_boxes.append(curr_item)
    return filtered_boxes


def main():
    image = load_img()
    image = thresholding(opening(to_grayscale(image)))
    cv2.imwrite("modified.jpg", image)
    chars, boxes = extract_bounding_boxes(image)
    detections = [Detections(c, b) for c, b in zip(chars, boxes)]
    detections = remove_false_positives(image, detections)
    detections = filter_overlapping_boxes(detections)
    sorted_detections = sort_top_left(detections)
    image = draw_bounding_boxes(image, [d.box for d in sorted_detections])
    # for c, b in sorted_items:
    #     image = draw_bounding_boxes(image, [b])
    #     print("Drawing ", c, b)
    #     cv2.imshow("image", image)
    #     cv2.waitKey(0)

    cv2.imwrite("annotated.jpg", image)

    # i0 = sorted_items[6][1]
    # print(image.shape)
    # print(sorted_items[6][0], i0)

    # image_height = image.shape[0]
    # h = i0.top - i0.bottom
    # w = i0.right - i0.left
    # top = image_height - i0.top

    # new_image = image[top : top + h, i0.left : i0.left + w]
    # print(new_image)
    # cv2.imwrite("extracted_image.jpg", new_image)


main()

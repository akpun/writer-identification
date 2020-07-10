import os
import sys
from os.path import exists

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

path = sys.argv[1]
new_path = sys.argv[2]
dataset = sys.argv[3]


def smart_crop(img):
    # (1) Convert to gray, and threshold
    s = img.shape
    if len(s) > 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    th, threshed = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    # (2) Morph-op to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 200))
    morphed = cv2.morphologyEx(threshed, cv2.MORPH_CROSS, kernel)
    # (3) Find the max-area contour
    cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnt = sorted(cnts, key=cv2.contourArea)[-1]
    # (4) Crop and save it
    x, y, w, h = cv2.boundingRect(cnt)
    dst = img[y:y + h, x:x + w]
    return dst


def crop_iam(image):
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    result = image.copy()

    # gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Get horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (200, 1))
    remove_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(remove_horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # get coords of lines
    if len(cnts) > 3:
        _, y1, _, _ = cv2.boundingRect(cnts[2])
        _, y2, _, _ = cv2.boundingRect(cnts[0])
    else:
        _, y1, _, _ = cv2.boundingRect(cnts[1])
        _, y2, _, _ = cv2.boundingRect(cnts[0])

    # remove lines
    cv2.drawContours(result, cnts, -1, (255, 255, 255), 5)

    # crop
    result = result[y1:y2, 50:2400]
    return result


def crop_firemaker(image, crop):
    crop_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    crop_img = img[700:2800, :].copy()
    return crop_img


if __name__ == '__main__':
    if not exists(new_path):
        os.makedirs(new_path)

    print("Started.")
    for subdir, dirs, files in os.walk(path):
        for image in tqdm(files):
            name = os.path.join(subdir, image)
            if dataset == 'firemaker':
                crop_img = crop_firemaker(name)
            elif dataset == 'iam':
                crop_img = crop_iam(name)
            else:
                crop_img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
            img = smart_crop(crop_img)
            cv2.imwrite(os.path.join(new_path, image), img)
    print("Finished")

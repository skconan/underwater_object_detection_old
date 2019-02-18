#!/usr/bin/env python
"""
    File name: lib.py
    Author: skconan
    Date created: 2018/10/08
    Python Version: 3.5
"""


import constans as CONST
import numpy as np
import cv2 as cv

def get_kernel(shape='rect', ksize=(5, 5)):
    if shape == 'rect':
        return cv.getStructuringElement(cv.MORPH_RECT, ksize)
    elif shape == 'ellipse':
        return cv.getStructuringElement(cv.MORPH_ELLIPSE, ksize)
    elif shape == 'plus':
        return cv.getStructuringElement(cv.MORPH_CROSS, ksize)
    else:
        return None

def normalize(gray):
    return np.uint8(255 * (gray - gray.min()) / (gray.max() - gray.min()))

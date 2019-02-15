#!/usr/bin/env python
"""
    File name: my_bg_subtraction.py
    Author: skconan
    Date created: 2010/01/10
    Python Version: 3.5
"""


import constans as CONST
import numpy as np
import cv2 as cv
from lib import *

def neg_bg_subtraction():
    cap = cv.VideoCapture(CONST.PATH_VDO + r'\pool_gate_05.mp4')
    while cap.isOpened():
        _, bgr = cap.read()
        if bgr is None:
            continue
        bgr = cv.resize(bgr, (0, 0), fx=0.25, fy=0.25)

        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        bg = cv.medianBlur(gray, 61)
        fg = cv.medianBlur(gray, 5)

        sub_sign = np.int16(fg) - np.int16(bg)
        sub_pos = np.clip(sub_sign.copy(),0,sub_sign.copy().max())
        sub_neg = np.clip(sub_sign.copy(),sub_sign.copy().min(),0)

        sub_pos = normalize(sub_pos)
        sub_neg = normalize(sub_neg)

        cv.imshow('sub_pos',sub_pos)
        cv.imshow('sub_neg',sub_neg)

        _, obj = cv.threshold(
            sub_neg, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
        )

        cv.imshow('obj',obj)
        
        bg = np.uint8(bg)
        fg = np.uint8(fg)

        cv.imshow('original_bgr', bgr)
        cv.imshow('bg', np.uint8(bg))
        cv.imshow('fg', fg)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    neg_bg_subtraction()

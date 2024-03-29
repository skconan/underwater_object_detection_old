#!/usr/bin/env python
"""
    File name:gate_bg_subtraction.py
    Author: skconan
    Date created: 2018/10/14
    Python Version: 3.5
"""


import constans as CONST
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from statistic import Statistic
from fourier_transform import FourierTransform


def gate():
    cap = cv.VideoCapture(CONST.PATH_VDO + r'\pool_gate_05.mp4')
    min_gray = 120
    max_gray = 230
    ref = np.zeros((200, 200), np.uint8)
    ref[0:100, :] = min_gray
    ref[100:, :] = max_gray
    cv.imshow('ref', ref)
    while cap.isOpened():
        _, bgr = cap.read()
        if bgr is None:
            continue
        bgr = cv.resize(bgr, (0, 0), fx=0.25, fy=0.25)

        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        bg = cv.medianBlur(gray, 61)
        fg = cv.medianBlur(gray, 5)
        sub = np.uint8(abs(fg - bg))

        bg = np.uint8(bg)
        fg = np.uint8(fg)

        obj1 = np.zeros(gray.shape, np.uint8)
        obj2 = np.zeros(gray.shape, np.uint8)
        obj1[sub > min_gray] = 255
        obj2[sub < max_gray] = 255
        obj = cv.bitwise_and(obj1, obj2)

        cv.imshow('original_bgr', bgr)
        cv.imshow('bg', np.uint8(bg))
        cv.imshow('fg', fg)

        cv.imshow('sub', sub)

        cv.imshow('obj', obj)
        cv.imshow('obj1', obj1)
        cv.imshow('obj2', obj2)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        histr = cv.calcHist([sub], [0], None, [256], [0, 256])
        plt.plot(histr, color='red')
 
        plt.pause(0.0001)
        plt.clf()
    plt.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    stat = Statistic()
    gate()

#!/usr/bin/env python
"""
    File name:gate.py
    Author: skconan
    Date created: 2018/10/10
    Python Version: 3.5
"""


import constans as CONST
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from statistic import Statistic
from fourier_transform import FourierTransform


def gate():
    cap = cv.VideoCapture(CONST.PATH_VDO + r'\pool_gate_03.mp4')

    while cap.isOpened():
        _, bgr = cap.read()
        if bgr is None:
            continue
        bgr = cv.resize(bgr, (0, 0), fx=0.3, fy=0.3)
        hsv = cv.cvtColor(bgr.copy(), cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        bg = cv.medianBlur(gray, 31)

        scharr_y = cv.Scharr(gray, cv.CV_8U, 0, 1, scale=0.1)
        scharr_x = cv.Scharr(gray, cv.CV_8U, 1, 0, scale=0.1)
        scharr = cv.bitwise_or(scharr_y, scharr_x)

        kernel = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]])

        p99 = stat.get_percentile(scharr, 99)

        print(p99)
        _, scharr_th = cv.threshold(scharr, p99, 255, cv.THRESH_BINARY)

        cv.imshow('original_bgr', bgr)
        # cv.imshow('original_bg', bg)
        cv.imshow('gray', gray)
        # cv.imshow('sobel', sobel)
        cv.imshow('scharr', scharr)
        # cv.imshow('sobel_th', sobel_th)
        cv.imshow('scharr_th', scharr_th)

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        histr = cv.calcHist([scharr], [0], None, [256], [0, 256])
        plt.plot(histr, color='blue')
        plt.pause(0.001)
        plt.clf()
    plt.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    stat = Statistic()
    gate()

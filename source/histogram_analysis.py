#!/usr/bin/env python
"""
    File name: histogram_analysis.py
    Author: skconan
    Date created: 2018/10/08
    Python Version: 3.5
"""


import constans as CONST
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from statistic import Statistic
from fourier_transform import FourierTransform


def normalize(gray):
    min = stat.get_min(gray)
    max = stat.get_max(gray)
    print(stat.get_skewness(gray))
    gray = (gray - min) / (max - min)
    gray *= 200
    gray += 20
    gray = np.uint8(gray)
    return gray

def camera_test():
    cap = cv.VideoCapture(1)
    cap.set(3, 1936)
    cap.set(4, 1216)
    while cap.isOpened():
        ret, bgr = cap.read()
        if bgr is None:
            continue
        bgr = cv.resize(bgr,(0,0), fx=0.5, fy=0.5)
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        gray_norm = normalize(gray)
        print("Skewness:",stat.get_skewness(gray))
        print("MODE:", stat.get_mode(gray))
        print("Q0:", stat.get_quantile(gray,0))
        print("Q1:", stat.get_quantile(gray,1))
        print("Q2:", stat.get_quantile(gray,2))
        print("Q3:", stat.get_quantile(gray,3))
        print("Q4:", stat.get_quantile(gray,4))
        # fft_v = ft.image_to_fft(v)
        # fft_v = ft.shift_to_center(fft_v)
        # logz = ft.get_log_scale(fft_v)  
        # logz /= stat.get_max(logz)
        # logz *= 255
        # logz = np.uint8(logz)
        # logz[logz > 150] = 255
        cv.imshow('original', gray)
        # cv.imshow('logz', logz)
        cv.imshow('normalize', gray_norm)
        # cv.imshow('fft_v', np.uint8(fft_v))
        
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        histr = cv.calcHist([gray],[0],None,[256],[0,256])
        plt.plot(histr,color = 'blue')
        histr = cv.calcHist([gray_norm],[0],None,[256],[0,256])
        plt.plot(histr,color = 'red')
        # plt.xlim([0,256])
        # plt.plot(logz)
        plt.pause(0.05)
        plt.clf()
    plt.close()
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    stat =  Statistic()
    camera_test()
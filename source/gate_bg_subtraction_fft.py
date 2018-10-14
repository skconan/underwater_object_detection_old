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
    cap = cv.VideoCapture(CONST.PATH_VDO + r'\pool_gate_05.mp4')
    min_gray = 120
    max_gray = 240
    ref = np.zeros((100,100),np.uint8)
    ref[0:50,:] = min_gray    
    ref[50:,:] = max_gray
    cv.imshow('ref',ref)
    while cap.isOpened():
        try:
            _, bgr = cap.read()
        except:
            break
        if bgr is None:
            continue
        bgr = cv.resize(bgr, (0, 0), fx=0.25, fy=0.25)
   
        gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
        bg = cv.medianBlur(gray, 61)
        fg = cv.medianBlur(gray, 5) 

        # sub_res = np.zeros(gray.shape,np.uint8)
        # for i in range(gray.shape[0]):
        #     for j in range(gray.shape[1]):
        #         sub_res[i,j] = abs(fg[i,j] - bg[i,j])
        # sub = np.uint8(abs(fg - bg))
        
        # bg = np.uint8(bg)
        # fg = np.uint8(fg)

        fft_bg = fft.image_to_fft(bg)
        fft_fg = fft.image_to_fft(fg)

        fft_bg = fft.shift_to_center(fft_bg)
        fft_fg = fft.shift_to_center(fft_fg)

        fft_bg_log = fft.get_log_scale(fft_bg)
        fft_fg_log = fft.get_log_scale(fft_fg)

        fft_sub = fft_fg - fft_bg
        cv.imshow('log_sub', np.uint8(fft_sub.copy()))

        fft_sub = fft.fft_to_image(fft_sub)
        fft_sub = (fft_sub - fft_sub.min()) / (fft_sub.max() - fft_sub.min())
        fft_sub = fft_sub * 255
        fft_sub = np.uint8(fft_sub)
        fft_sub[fft_sub < 100] = 255
        # obj1 = np.zeros(gray.shape,np.uint8)
        # obj2 = np.zeros(gray.shape,np.uint8)
        # obj1[sub > min_gray] = 255 
        # obj2[sub < max_gray] = 255 
        # obj = cv.bitwise_and(obj1,obj2)


        cv.imshow('original_bgr', bgr)
        cv.imshow('bg', np.uint8(bg))
        cv.imshow('fg', fg)
        cv.imshow('log_bg', fft_bg_log)
        cv.imshow('log_fg', fft_fg_log)
        cv.imshow('fft_sub', fft_sub)
        # cv.imshow('sub', sub)
  
        # cv.imshow('obj', obj)
        # cv.imshow('obj1', obj1)
        # cv.imshow('obj2', obj2)

  

        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        # histr = cv.calcHist([fg], [0], None, [256], [0, 256])
        # plt.plot(histr, color='red')
        # histr = cv.calcHist([bg], [0], None, [256], [0, 256])
        # plt.plot(histr, color='green')
        # histr = cv.calcHist([sub], [0], None, [256], [0, 256])
        # plt.plot(histr, color='blue')
    #     plt.pause(0.0001)
    #     plt.clf()
    # plt.close()
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    fft = FourierTransform()
    stat = Statistic()
    gate()

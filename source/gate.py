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
        bgr = cv.resize(bgr,(0,0), fx=0.3, fy=0.3)
        hsv = cv.cvtColor(bgr.copy(),cv.COLOR_BGR2HSV)
        gray = cv.cvtColor(bgr,cv.COLOR_BGR2GRAY)
        bg = cv.medianBlur(gray,31)
  
        # gray = cv.blur(gray,(7,7))
        # sobel_y = cv.Sobel(gray,cv.CV_8U,0,1,ksize=3,scale=0.5)
        # sobel_x = cv.Sobel(gray,cv.CV_8U,1,0,ksize=3,scale=0.5)
        # sobel = cv.bitwise_or(sobel_y, sobel_x)

        scharr_y = cv.Scharr(gray,cv.CV_8U,0,1,scale=0.1)
        scharr_x = cv.Scharr(gray,cv.CV_8U,1,0,scale=0.1)
        scharr = cv.bitwise_or(scharr_y, scharr_x)

        kernel = np.array ([
            [1,1,1],
            [1,1,1],
            [1,1,1]])

        # max_scharr = stat.get_max(scharr)
        # mean_scharr = stat.get_mean(scharr)
        # q3_sobel = stat.get_quantile(sobel,3.9)
        # print(q3_sobel)
        # print(q3_scharr)
        # print(max_scharr, mean_scharr)
        p99 = stat.get_percentile(scharr,99)
        # th_value = (max_scharr + mean_scharr) / 2.
        print(p99)
        _, scharr_th = cv.threshold(scharr,p99,255,cv.THRESH_BINARY)
        # _, sobel_th = cv.threshold(sobel,q3_sobel,255,cv.THRESH_BINARY)
        
        # scharr_dilate = cv.dilate(scharr.copy(),kernel,iterations=1)
        # sobel_dilate = cv.dilate(sobel.copy(),kernel,iterations=1)
        
        # result = cv.bitwise_and(scharr_th, sobel_th)

        # median_scharr = cv.medianBlur(scharr.copy(),5)
        # _, mask_median_scharr = cv.threshold(median_scharr.copy(), 20, 255, cv.THRESH_BINARY)
        # scharr_erode = cv.erode(scharr.copy(),kernel)
        # scharr_dilate = cv.dilate(scharr_erode,kernel,iterations=3)
        # _, mask_median_sobely = cv.threshold(sobely.copy(),1,255,cv.THRESH_BINARY)
        # sobely_dilate = cv.dilate(mask_median_sobely, kernel, iterations=3)


        cv.imshow('original_bgr', bgr)     
        # cv.imshow('original_bg', bg)     
        cv.imshow('gray', gray)
        cv.imshow('obj', obj)
        # cv.imshow('sobel', sobel)
        cv.imshow('scharr',scharr)
        # cv.imshow('sobel_th', sobel_th)
        cv.imshow('scharr_th',scharr_th)
        # cv.imshow('result',result)
        # cv.imshow('scharr_erode',scharr_erode)
        # cv.imshow('scharr_dilate',scharr_dilate)
        # cv.imshow('sobely_dialte', sobely_dilate)
        # cv.imshow('median_scharr',median_scharr)
        # cv.imshow('mask_median_scharr',mask_median_scharr)
        # cv.imshow('mask_median_sobely',mask_median_sobely)
        # cv.imshow('result',mask_median_scharr - sobely_dilate)
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
        histr = cv.calcHist([scharr],[0],None,[256],[0,256])
        plt.plot(histr,color = 'blue')
        # histr = cv.calcHist([sobel],[0],None,[256],[0,256])
        # plt.plot(histr,color = 'red')
        # plt.xlim([0,256])
        # plt.plot(logz)
        plt.pause(0.001)
        plt.clf()
    plt.close()
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    stat =  Statistic()
    gate()
#!/usr/bin/env python
"""
    File name: histogram_matching.py
    Author: skconan
    Date created: 2018/10/08
    Python Version: 3.5
"""


import constans as CONST
import numpy as np
import cv2 as cv

def normalize(gray):
    gray = (gray - gray.min()) / (gray.max() - gray.min())
    gray *= 150
    gray += 50
    gray = np.uint8(gray)
    return gray

def img_test():
    gray = cv.imread(CONST.PATH_IMAGE+r'\Screenshot (108).png', 0)
    gray = cv.resize(gray,(0,0), fx=0.5, fy=0.5)
    norm_gray = normalize(gray)
    cv.imshow('original', gray)
    cv.imshow('normalize', norm_gray)
    cv.waitKey(-1)
    cv.destroyAllWindows()

def vdo_test():
    cap = cv.VideoCapture(CONST.PATH_VDO + r'\robosub_12.mp4')
    while cap.isOpened():
        ret, gray = cap.read()
        if gray is None:
            continue
        # gray = cv.imread(CONST.PATH_IMAGE+r'\Screenshot (108).png', 0)
        gray = cv.cvtColor(gray, cv.COLOR_BGR2GRAY)
        gray = cv.resize(gray,(0,0), fx=0.5, fy=0.5)
        norm_gray = normalize(gray)
        cv.imshow('original', gray)
        cv.imshow('normalize', norm_gray)
        
        k = cv.waitKey(1) & 0xff
        if k == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    vdo_test()
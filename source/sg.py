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
    # cap = cv.VideoCapture(CONST.PATH_VDO + r'\pool_gate_05.mp4')
    # cap = cv.VideoCapture(CONST.PATH_VDO + r'\pool_gate_09.mp4')
    # cap = cv.VideoCapture(CONST.PATH_VDO + r'\pool_gate_12.mp4')
    
    bgr = cv.imread(r"C:\Users\skconan\Desktop\underwater_object_detection\images\sg.png")
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    lower = np.array([0,0,0],np.uint8)
    upper = np.array([179,255,70],np.uint8)
    mask_hsv = cv.inRange(hsv, lower, upper)
    # _, bgr = cap.read()
    # if bgr is None:
    #     continue
    # bgr = cv.resize(bgr, (0, 0), fx=0.25, fy=0.25)
    gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)
    # bg = cv.medianBlur(gray, 61)
    # fg = cv.medianBlur(gray, 5)
    # bg = cv.blur(gray,(61,61))
    bg = cv.blur(gray,(31,31))
    # fg = cv.blur(gray,(5,5))
    fg = cv.blur(gray,(3,3))
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
    kernel = get_kernel('plus',(3,3))
    kernel = get_kernel('plus',(3,3))
    obj_erode = cv.erode(obj.copy(),kernel)
    mask = cv.dilate(obj.copy(),kernel)
    contours, _ =  cv.findContours(mask_hsv.copy(),cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        x,y,w,h = np.int0(cv.boundingRect(cnt))
        if w < 10 or h < 10:
            continue
        (x,y),r = cv.minEnclosingCircle(cnt)
        area_cir = 3.14*r*r
        area_cnt = cv.contourArea(cnt)
        if area_cir / area_cnt > 1.25:
            continue
        cv.circle(bgr, (int(x), int(y)), int(r), (255,0,255),4)
        # cv.rectangle(bgr,(x,y),(x+w,y+h),(0,0,255),2)
        mask_obj = np.zeros(gray.shape,np.uint8)
        # cv.rectangle(mask_obj,(x,y),(x+w,y+h),(255),-1)
        # cv.imshow('mask',mask_obj.copy())
        mask_obj = cv.bitwise_and(mask.copy(),mask_obj.copy())
        # cv.imshow('mask_obj',mask_obj.copy())
        # gray_input = cv.bitwise_and(gray.copy(),gray.copy(),mask=mask_obj)
        # cv.imshow('gray_input',gray_input.copy())
        # gray_input = s.resize_image(gray_input)
        # one_hot = s.img_to_onehot(gray_input)
        # predict = model.predict(one_hot)
        # predict = np.argmax(predict,axis=1)
        # # print(predict)
        # if predict[0] == 1:
        #     cv.rectangle(bgr,(x,y),(x+w,y+h),(0,255,255),2)
        # elif predict[0] == 0:
        #     cv.rectangle(bgr,(x,y),(x+w,y+h),(255,0,0),2)
    bg = np.uint8(bg)
    fg = np.uint8(fg)
    cv.imshow('original_bgr', bgr)
    # cv.imshow('bg', bg)
    # cv.imshow('fg', fg)
    cv.imshow('obj',obj)
    cv.imshow('hsv',hsv)
    cv.imshow('mask_hsv',mask_hsv)
    cv.imshow('obj_erode',obj_erode)
    k = cv.waitKey(-1) & 0xff
    # if k == ord('q'):
    #     break
    # cap.release()
    # cv.destroyAllWindows()


if __name__ == '__main__':
    neg_bg_subtraction()

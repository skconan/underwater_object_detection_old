#!/usr/bin/env python
import cv2 as cv
import rospy
import numpy as np
from sensor_msgs.msg import CompressedImage
import math
import dynamic_reconfigure.client
import constant as CONST
from vision_lib import *
from matplotlib import pyplot as plt


class AutoExposure:

    def __init__(self, sub_topic, client_name, ev_default=1, ev_min=0.5, camera_position='front'):
        print_result("init_node_auto_exposure")
        self.hsv = None
        self.image = None
        self.sub_topic = sub_topic
        self.client_name = client_name
        self.ev_default = ev_default
        self.ev_min = ev_min
        self.subsampling = 0.5
        self.sub_image = rospy.Subscriber(
            sub_topic, CompressedImage, self.img_callback,  queue_size=10)
        self.client = dynamic_reconfigure.client.Client(self.client_name)
        print_result('set_client')
        self.set_param('exposure', self.ev_default)

        '''
            PID CONTROL
        '''
        print_result('Initial PID value')
        self.KP = 0
        self.KI = 0
        self.KD = 0

        self.err = 0
        self.previous_err = 0
        self.sum_err = 0
        self.diff_err = 0 
        
        self.expected_v = 10

    def get_mode(self, data):
        if len(data.shape) > 1:
            data = data.ravel()
        count = np.bincount(data)
        max = count.max()
        count = list(count)
        return count.index(max)

    def img_callback(self, msg):
        arr = np.fromstring(msg.data, np.uint8)
        self.image = cv.resize(cv.imdecode(
            arr, 1), (0, 0), fx=self.subsampling, fy=self.subsampling)
        self.hsv = cv.cvtColor(self.image, cv.COLOR_BGR2HSV)

    def set_param(self, param, value):
        value = max(self.ev_min, value)
        params = {str(param): value}
        config = self.client.update_configuration(params)
        print_result('set_param: '+str(param)+' '+ str(value))

    def get_param(self, param):
        value = rospy.get_param("/" + str(self.client_name) + '/'+ str(param), None)
        return value

    def nothing(self,x):
        pass

    def adjust_exposure_time(self):
        cv.namedWindow('PID Tuning')
        cv.createTrackbar('KP','PID Tuning',0,255,self.nothing)
        cv.createTrackbar('KI','PID Tuning',0,255,self.nothing)
        cv.createTrackbar('KD','PID Tuning',0,255,self.nothing)

        cv.setTrackbarPos('KP','PID Tuning',0)
        cv.setTrackbarPos('KI','PID Tuning',0)
        cv.setTrackbarPos('KD','PID Tuning',0)
        
        r = rospy.Rate(15)

        while not rospy.is_shutdown():
            if self.hsv is None:
                print_result('image is none')
                continue

            h, s, v = cv.split(self.hsv)
            v_one_d = v.ravel()
            v_mode = self.get_mode(v_one_d)

            ev = self.get_param('exposure')
            print_result('Real Exposure Value')
            print_result(ev)

            ############# DISPLAY #############
            cv.imshow('PID Tuning', self.image)
            cv.waitKey(1)
            plt.hist(v_one_d, 256, [0, 256])
            plt.pause(0.001)
            plt.cla()
            ###################################

            if ev is None:
                print_result('EV is None')
                continue

            KP = cv.getTrackbarPos('KP','PID Tuning')
            KI = cv.getTrackbarPos('KI','PID Tuning')
            KD = cv.getTrackbarPos('KD','PID Tuning')

            self.KP = KP / 100.
            self.KI = KI / 100.
            # self.KI = 0
            self.KD = KD / 100.

            ############# PID #############
            self.err = self.expected_v - v_mode
            self.sum_err += ((self.err + self.previous_err) / 2 )*  0.5 
            self.diff_err = (self.err - self.previous_err) / 0.5 

            print('KP:', self.KP)
            print('KI:', self.KI)
            print('KD:', self.KD)

            print('ERR:', self.err)
            print('PRV_ERR:', self.previous_err)
            print('SUM_ERR:', self.sum_err)

            ev += (self.KP * self.err) + (self.KI * self.sum_err) + (self.KD * self.diff_err)

            self.previous_err = self.err

            ###################################

            print_result('Exposure')
            print_result(ev)
            print_result('V mode')
            print_result(v_mode)

            self.set_param('exposure', ev)
            print('EV: ',ev)
            rospy.sleep(0.5)

if __name__=='__main__':
    rospy.init_node('Auto_Exposure_PID')
    AE = AutoExposure('/front/image_raw/compressed','ueye_cam_nodelet_front')
    AE.adjust_exposure_time()
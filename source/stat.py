#!/usr/bin/env python
"""
    File name: statistics.py
    Author: skconan
    Date created: 2018/10/08
    Python Version: 3.5
"""
import numpy as np


class Statistics():
    def __init__(self):
        pass
    
    def convert_to_oneD(self, data):
        if len(data.shape) == 2:
            return data.ravel()
        return data
    
    def convert_to_np(self,data):
        return np.array(data)
    
    def get_mode(self, data):
        data = self.convert_to_oneD(data)
        count = np.bincount(data)
        max = count.max()
        count = list(count)
        return count.index(max)

    def get_std(self, data):
        data = self.convert_to_oneD(data)
        return np.std(data)

    def get_max(self, data):
        return self.convert_to_np(data).max()

    def get_min(self, data):
        return self.convert_to_np(data).min()

    def get_range(self, data):
        return self.get_max(data) - self.get_min(data)


    
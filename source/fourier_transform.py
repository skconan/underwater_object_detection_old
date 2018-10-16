#!/usr/bin/env python
"""
    File name: fourier_transform.py
    Author: skconan
    Date created: 2018/10/08
    Python Version: 3.5
"""

import numpy as np


class FourierTransform:
    def __init__(self):
        self.fft = np.fft
        

    def image_to_fft(self, grayscale):
        return self.fft.fft2(grayscale)

    def fft_to_image(self, image_fft2):
        return self.fft.ifft2(image_fft2)

    def shift_to_center(self, image_fft2):
        return self.fft.fftshift(image_fft2)

    def shift_to_corner(self, image_fft2):
        return self.fft.ifftshift(image_fft2)

    def get_magnitude(self, image_fft2):
        return np.abs(image_fft2)

    def get_log_scale(self, image_fft2):
        magnitude = self.get_magnitude(image_fft2)
        logz = np.log(1+magnitude)
        logz = np.uint8((logz - logz.min())*255/(logz.max()-logz.min()))
        return logz
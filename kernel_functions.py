#!/usr/bin/env pypy3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 15:54:22 2018

@author: Rasit ULUG
Middle East Technical University
"""

import numpy as np


"""
This file calculate:
    Gaussian Kernel
    Thin Plate splines Kernel
    
"""

def gaussian_kernel(dist,shape):
    return np.exp(-(shape**2 * dist**2))

def thin_plate_spline(dist,shape):
    return dist**2 * np.log(dist)
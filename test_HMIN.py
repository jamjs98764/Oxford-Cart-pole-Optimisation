# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 15:57:57 2018

@author: jianhong
"""
import numpy as np
import HMIN 

d = 2
a = -1. * np.ones(d)
b = 2. * np.ones(d)
L = 1.
p = 1.
alpha = 0.
minbnd = 1

def fct(x):
    return np.amax(abs(x)) + 1

hopt2 = HMIN.HMIN_multidim(a,b,fct,L,p)
global t

argmin, m, i, counter = HMIN.minimiseUntilErrthresh_(hopt2, 0.005)

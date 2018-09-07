# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 19:13:58 2018

@author: jianhong
"""

upp = 0
list = np.arange(50,480)
for n in list:
    cur_x = s_tex[n]
    cur_y = f_tex[n]
    sample_s_old = s_tex[0:n-1]
    sample_f_old = f_tex[0:n-1]
    
    X = np.array([cur_x for col in range(n-1)])
    
    m_rowvec = vecmetric_maxnorm(X, sample_s_old,1)
    F = np.array([cur_y for col in range(n-1)])
    
    diffs_f = vecmetric_2norm(F, sample_f_old,1.0) - epsilon*2
    
    a = max(diffs_f/(m_rowvec**LACKI.p))
    upp = max(a,upp)
print(upp)
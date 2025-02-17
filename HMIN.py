# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:16:47 2018

@author: jianhong
"""
##### Importing packages

import numpy as np
import random
import copy

##### Sample data structures

class sample():
    def __init__(self, inp, outp, errbnd):
        self.inp= inp
        self.outp = outp
        self.errbnd = errbnd

class HMIN_multidim():#Type for online minimisation of an L-p Hoelder function
    #where the L-P Hoelderness is given relative to the maximum-norm
    def __init__(self, a, b, fct, L, p = 1.0, alpha = 0.):
        self.a= a
        self.b = b
        self.fct = fct
        self.L = L # Hoelder constant est
        self.p = p # Hoelder exponent est
        D = sample(np.array([(a+b)/2]), np.array([fct((a+b)/2)]), 0.)
        self.D = D # array of samples
        self.alpha = alpha # threshold for Lip const est
        gridradii, rightradii = compute_radii_of_sample_grid(D.inp, a, b) # right radii is redundant so not stored
        self.gridradii = gridradii # matrix, radii of the input grid, d x number_samples
        minbnd, indminbnd, errbnds = comp_minbnd(L, p, D.outp, gridradii[0]) # integer, index of minimum value of sample hyperrect where minbnd occurs
        self.minbnd = minbnd # min value of floor


def rem_sample_points_(sample, inds):
    # Removes one sample point from sample matrix based on inds given
    # inds must be list
    ss = sample.inp.shape[0] # rows are number of samples
    
    for ind in inds:
        indkeep = [i for i in range(ss)].pop(ind)
        
    sample.inp = sample.inp[indkeep] # keep samples which are in indkeep
    sample.outp = sample.outp[indkeep]
    return None

def compute_radii_of_sample_grid(inp, minxvec = [], maxxvec = []):
    """
    #inp[:,j] \leq minxvec componentwise
    #inp[:,j] \leq maxxvec componentwise
    # return ra,rb
    #ra[i,j] = radius in left(neg) direction from sample s_j along dimension i
    #rb[i,j] = radius in right direction from sample s_j along dimension i
    """
    m, d = inp.shape

    if m < 1:
        return []

    if minxvec == []:
        minxvec = np.amin(inp, axis = 1)
    
    if maxxvec == []:
        maxxvec = np.amax(inp, axis = 1)
    ri = np.zeros((m+1,d)) #radii of all samples in ith dimension, ri[i,j] is left radius for jth sample in ith dim
    #similarly, ri[i,j+1] is the right radius of the jth sample in ith dim
    for i in range(d):
        v = np.insert(inp[:,i], 0, minxvec[i])
        v = np.append(v,maxxvec[i])
        v = np.sort(v)
        v2 = v[1:]
        ri[:,i] = abs(v2-v[0:-1])/2
        ri[0,i] = 2*ri[0,i]
        ri[-1,i] = 2*ri[-1,1] 
    
    ra = ri[0:-1,:]
    rb = ri[1:,:]
    
    return ra, rb # TODO: these are redundant, which is why we only store one

def find_min(obj):
    # Where obj is type HMIN_multidim
    m = np.amin(obj.D.outp) # minimum value
    i = np.argmin(obj.D.outp) # index
    return m, i

def comp_minbnd(L, p, outp, gridradii):
    """
    L - float
    p - float
    outp - matrix, Rows are samples, samples are vectors
    gridradii - matrix, radii of the input grid, d x number_samples
    """
    if gridradii.ndim == 1:
        rmx = gridradii
    else:
        rmx = np.amax(gridradii, axis = 1)
    
    errbnds = L * (rmx**p)
    minbnds = outp - errbnds
    m = np.amin(minbnds) # minimum value
    i = np.argmin(minbnds) # index
    return m, i, errbnds[i]

def comp_minbnd_(obj):
    # Where obj is type HMIN_multidim
    obj.minbnd, obj.indminbnd, obj.errfvalmin = comp_minbnd(obj.L, obj.p, obj.D.outp, obj.gridradii)
    return None

##### Grid refinement

def SplitHyperrectAlongDim(c,r,m):
    if len(c) > 1:
        c1 = copy.deepcopy(c) # Assignment by value, not reference
        c3 = copy.deepcopy(c)
        rcp = r[:]
        c1[m] = c[m] - (r[m]*2/3)
        c3[m] = c[m] + (r[m]*2/3)
        rcp[m] = r[m] * 1/3
    else:
        c1 = c - (r*2/3)
        c3 = c+ (r*2/3)
        rcp = r *1/3   
    return (c1,c,c3), (rcp,rcp,rcp)

def SelectHyperrect2Split_(obj, method = 'minbnd'):
    # Where obj is type HMIN_multidim
    if method == 'minbnd':
        rmx = np.transpose(np.array([np.amax(obj.gridradii, axis = 1)]))
        errbnds = obj.L * (rmx**obj.p)
        minbnds = obj.D.outp - errbnds
        obj.minbnd = np.amin(minbnds) # minimum value
        #print('ind: ' + str(np.argmin(minbnds)))
        #print('minbnd: ' + str(obj.minbnd))
        #print('shape1: ' + str(obj.D.outp.shape))
        #print('shape2: ' + str(rmx.shape))
        obj.indminbnd = np.argmin(minbnds) # index
        return obj.indminbnd
    elif method == 'rndhyperrect':
        range = obj.gridradii.shape[1]
        return random.randint(0,range)
    else:
        return []
    

def RefineGrid_(obj, fct = [], method = 'minbnd'):
    #fct: function to be estimated/integrated/sampled
    #l = len(grid.r) # last element at l-1
    if fct == []:
        fct = obj.fct

    ind = SelectHyperrect2Split_(obj, method)
    c = obj.D.inp[ind]
    r = obj.gridradii[ind] 
    mx = np.amax(r) # minimum value, dim with maximum radius
    m = np.argmax(r) # index
    c_list, r_list = SplitHyperrectAlongDim(c,r,m)
    f_list = np.array([fct(c_list[0]), fct(c_list[2])])
        
    for element in f_list:
        obj.D.outp = np.vstack((obj.D.outp,element))
    obj.gridradii[ind] = r_list[1]
    obj.gridradii = np.vstack((obj.gridradii, r_list[0], r_list[2]))
    
    obj.D.inp = np.vstack((obj.D.inp, c_list[0], c_list[2]))    

    
def RefineGridNoOfTimes_(obj, no_times, fct = [], method = 'minbnd'):
    for i in range(no_times):
        RefineGrid_(obj, fct, method)

def minimiseUntilErrthresh_(obj, errthresh, maxiter = 1000, fct = [], method = 'minbnd'):
    m,i = find_min(obj)
    counter = 0
    
    while (abs(m-obj.minbnd) >= errthresh) & (counter <= maxiter):
        RefineGrid_(obj, fct, method)
        counter += 1
        m,i = find_min(obj)
    
    argmin = obj.D.inp[i]
    return argmin, m, i, counter


    
        
        
    
    





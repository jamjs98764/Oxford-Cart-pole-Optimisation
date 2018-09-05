# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 11:16:47 2018

@author: jianhong
"""
##### Importing packages

import numpy as np
import random


##### Sample data structures

sample = {}
sample['inp'] = np.array([[1,2],[3,4],[5,6]]) # array of arrays. Rows are samples, samples are vectors
sample['outp'] = np.array([[1,2],[3,4],[5,6]])
sample['errbnd'] = 0.1


HMIN_multidim = {} #Type for online minimisation of an L-p Hoelder function
    #where the L-P Hoelderness is given relative to the maximum-norm
HMIN_multidim['a'] = np.array([1,2]) # vector, I = {x : a <= x <= b componentwise}
HMIN_multidim['b'] = np.array([1,2]) # vector
HMIN_multidim['fct'] = list() # Function
HMIN_multidim['L'] = 0.1 # Hoelder constant est
HMIN_multidim['p'] = 0.1 # Hoelder exponent est
HMIN_multidim['D'] = sample # Dict "sample"
HMIN_multidim['alpha'] = 0.1 # threshold for Lip const est
HMIN_multidim['gridradii'] = np.array([[1,2],[3,4],[5,6]]) # matrix, radii of the input grid, d x number_samples
HMIN_multidim['indminbnd'] = 4 # integer, index of minimum value of sample hyperrect where minbnd occurs
HMIN_multidim['minbnd'] = 0.4 # min value of floor


def rem_sample_points_(sample, inds):
    # Removes one sample point from sample matrix based on inds given
    # inds must be list
    ss = sample['inp'].shape[0] # rows are number of samples
    
    for ind in inds:
        indkeep = [i for i in range(ss)].pop(ind)
        
    sample['inp'] = sample['inp'][indkeep] # keep samples which are in indkeep
    sample['outp'] = sample['outp'][indkeep]
    return None

def compute_radii_of_sample_grid(inp, minxvec = [], maxxvec = []):
    """
    #inp[:,j] \leq minxvec componentwise
    #inp[:,j] \leq maxxvec componentwise
    # return ra,rb
    #ra[i,j] = radius in left(neg) direction from sample s_j along dimension i
    #rb[i,j] = radius in right direction from sample s_j along dimension i
    """
    d, m = inp.shape
    if m < 1:
        return []
    
    if minxvec == []:
        minxvec = min(1, inp.all()) # TODO: all inp?
    
    if maxxvec == []:
        maxxvec = max(1, inp.all())
    
    ri = np.zeros((d,m+1)) #radii of all samples in ith dimension, ri[i,j] is left radius for jth sample in ith dim
    #similarly, ri[i,j+1] is the right radius of the jth sample in ith dim
    
    for i in range(d):
        v = [minxvec[i], inp[i], maxxvec[i]] # TODO check code with Jan
        v.sort
        v2 = v[1:]
        ri[i,:] = abs(v2-v[0:-1])/2
        ri[i,0] = 2*ri[i,0]
        ri[i,-1] = 2*ri[0,-1] # TODO check code with Jan
    
    ra = ri[:,0,-1]
    rb = ri[:,1:]
    
    return ra, rb

def find_min(obj):
    # Where obj is type HMIN_multidim
    m = np.amin(HMIN_multidim['D']['outp']) # minimum value
    i = np.argmin(HMIN_multidim['D']['outp']) # index
    return m, i

def comp_minbnd(L, p, outp, gridradii):
    """
    L - float
    p - float
    outp - matrix, Rows are samples, samples are vectors
    gridradii - matrix, radii of the input grid, d x number_samples
    """
    rmx = np.amax(gridradii, axis = 0)
    errbnds = L * (rmx**p)
    minbnds = outp - errbnds
    m = np.amin(minbnds) # minimum value
    i = np.argmin(minbnds) # index
    return m, i, errbnds[i]

def comp_minbnd_(obj):
    # Where obj is type HMIN_multidim
    obj['minbnd'], obj['indminbnd'], obj['errfvalmin'] = comp_minbnd(obj['L'], obj['p'], obj['D']['outp'], obj['gridradii'])
    return None

    
##### Grid refinement

def SplitHyperrectAlongDim(c,r,m):
    if len(c) > 1:
        c1 = c[:] # Assignment by value, not reference
        c3 = c[:]
        rcp = r[:]
        c1[m] = c[m] - (r[m]*2/3)
        c3[m] = c[m] - (r[m]*2/3)
        rcp[m] = r[m] * 1/3
    else:
        c1 = c - (r*2/3)
        c3 = c+ (r*2/3)
        rcp = r *1/3
    
    return (c1,c,c3), (rcp,rcp,rcp)

def SelectHyperrect2Split_(obj, method = 'minbnd'):
    # Where obj is type HMIN_multidim
    if method == 'minbnd':
        rmx = np.amax(obj['gridradii'], axis = 0)
        errbnds = obj['L'] * (rmx**obj['p'])
        minbnds = obj['D']['outp'] - errbnds
        obj['minbnd'] = np.amin(minbnds) # minimum value
        obj['indminbnd'] = np.argmin(minbnds) # index
        return obj['indminbnd']
    elif method == 'rndhyperrect':
        range = obj['gridradii'].shape[1]
        return random.randint(0,range)
    else:
        return []
    

def RefineGrid_(obj, fct = [], method = 'minbnd'):
    #fct: function to be estimated/integrated/sampled
    #l = len(grid.r) # last element at l-1
    if fct == []:
        fct = obj['fct']
    
    ind = SelectHyperrect2Split_(obj, method)
    c = obj['D']['inp'][:,ind]
    r = obj['gridradii'][:,ind] 
    mx = np.amin(r) # minimum value, dim with maximum radius
    m = np.argmin(r) # index
    
    c_list, r_list = SplitHyperrectAlongDim(c,r,m)
    
    f_list = [fct(c_list)[0], fct(c_list)[2]]
    obj['D']['outp'] = [obj['D']['outp'], f_list]
    obj['gridradii'][:,ind] = r_list[1]
    obj['gridradii'] = [[obj]['gridradii'], r_list[0], r_list[2]]
    obj['D']['inp'] = [obj['D']['inp'], c_list[0], r_list[2]]
    

    
def RefineGridNoOfTimes_(obj, no_times, fct = [], method = 'minbnd'):
    for i in range(no_times):
        RefineGrid_(obj, fct, method)

def minimiseUntilErrthresh_(obj, errthresh, maxiter = 1000000, fct = [], method = 'minbnd'):
    m,i = find_min(obj)
    counter = 0
    
    while (abs(m-obj['minbnd']) >= errthresh) & (counter <= maxiter):
        RefineGrid_(obj, fct, method)
        counter += 1
        m,i = find_min(obj)
    
    argmin = obj['D']['inp'][:,i]
    return argmin, m, i, counter


    
        
        
    
    





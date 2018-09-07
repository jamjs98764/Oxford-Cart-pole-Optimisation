# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 09:37:45 2018

@author: jianhong
"""
import numpy as np
import matplotlib.pyplot as plt
import HPA

# ----- Begin: Parameters of the test run -----------

def f(x):
    return np.sin(abs(x[0] - 0.5)) + 1 #function defined so that only the first input dimension matters

L=1.
cond_dat_size = 5
partrain_dat_size = 5
epsilon = 0. #obs noise
hoelexp = 1.
d = 4 #dimensionality of the input space
par_init_guess = np.ones(d)
maxevals = 20000 #max. number of evaluations in parameter training
parmin = 0.001 * np.ones(d) # guess of minimal parameter (weight of dim)
parmax = 1.* np.ones(d) # guess of maximal parameter for each inp dim

# ----- End: Parameters of the test run -----------

# ----- Begin: data generation -----------

x = np.random.rand(1000,d) # rows are samples, d is vector-dimension
fx = f(x)

#use some data to condition KI on:

s_size = x.shape[0]
    
s_cond = x[np.random.permutation(s_size)[0:cond_dat_size]] #pick a few random sample points
fs_cond = f(s_cond)

D_cond= HPA.sample(s_cond,fs_cond,epsilon)

#use some separate data to as test set for training metric hyperparameters:
s_test_partrain = x[np.random.permutation(s_size)[0:partrain_dat_size]]#pick a few random sample points
fs_test_partrain = f(s_test_partrain)

# Now merge all available data together
D_cond_full = HPA.sample(np.vstack((s_cond, s_test_partrain)),
                          np.vstack((fs_cond, fs_test_partrain)),
                          epsilon)
# ----- End: data generation -----------

# ------------ creation of the KI learner based on D_cond_full and false frequency guess:


ki0 = HPA.HoelParEst(L, hoelexp, D_cond_full, HPA.vecmetric_maxnorm_ARD,
                     HPA.vecmetric_2norm, 0., par_init_guess, np.array([1.]))

# ------------ creation of the KI learner only based on D_cond, but with trained and false frequency guess:
ki1_VS = HPA.HoelParEst(L, hoelexp, D_cond, HPA.vecmetric_maxnorm_ARD, 
                        HPA.vecmetric_2norm, 0., par_init_guess, np.array([1.]))

Lparopt = ki1_VS.L 

HPA.KI_optimise_pars_metric_inp_testdat_VS_(ki1_VS, parmin,parmax, s_test_partrain, 
                                                  fs_test_partrain, Lparopt, maxevals)
print("ki1_VS: Parameter found:" + str(ki1_VS.pars_metric_inp))
ki1_VS.D = D_cond_full


ki1 = HPA.HoelParEst(L, hoelexp, D_cond, HPA.vecmetric_maxnorm_ARD,
                     HPA.vecmetric_2norm, 0., par_init_guess, np.array([1.]))
Lparopt = ki1.L


HPA.KI_optimise_pars_metric_inp_testdat_(ki1, parmin,parmax, s_test_partrain,
                                         fs_test_partrain, Lparopt, maxevals)
print("ki1: Parameter found: " + str(ki1.pars_metric_inp))
ki1.D = D_cond_full






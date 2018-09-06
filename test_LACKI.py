# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 10:10:27 2018

@author: jianhong
"""

# import GPy
import matplotlib.pyplot as plt
import numpy as np
import HPA
import tictoc


##### Parameters

outputdim =1 #set to 1 .. can be set to different vals if data has multiple output dimensions
L=1. #should be set to 1
epsilon =0.#assumed obs noise

hoelexp = .7
#d = 1 #dimensionality of the input space
par_init_guess = np.array([1.])
maxevals = 4000#max. number of evaluations in parameter training
Lmin = 0.
Lmax = 10.

d =1 #input space dimensionality

inp_weights = 1./np.arange(1,d+1)

test_set_size = 250
#tex_percentage = 2 #percentage of sample used for training
cond_percentage = 50

numlearners = 2
numtestruns = 1

# ---- let's go....
T_pred_hist =  np.zeros([numtestruns,numlearners])
T_train_hist =  np.zeros([numtestruns,numlearners])
mean_err_hist =  np.zeros([numtestruns,numlearners])
std_err_hist =  np.zeros([numtestruns,numlearners])
mean_test_err_hist =  np.zeros([numtestruns,numlearners])
median_test_err_hist =  np.zeros([numtestruns,numlearners])
median_err_hist = np.zeros([numtestruns,numlearners])


tex_set_size  = 500

print("\n")
# ============= Start:Data extraction and preparation  =====

global inp_test_set 
inp_test_set = np.random.rand(test_set_size, d)
global inp_tex_set 
inp_tex_set = np.random.rand(tex_set_size, d)

#PyPkgfun = x -> minimum(repmat(inp_weights,1,size(x)[2]).*(abs(-cos(2*π *x))+x),1)
#fun = x -> mean(repmat(inp_weights,1,size(x)[2]).*(abs(-cos(2*π *x))+x),1)
#fun = x ->(abs(-cos(2*π *x)))[1,:]
#fun = x ->mean(abs(-cos(2*π *HPA.vecmetric_2norm(0*x,repmat(inp_weights,1,size(x)[2]).*x))),1)
#fun = x -> mean(abs(-cos(2*π *x))+x,1)
#fun = x -> mean(abs(-cos(2*π *repmat(inp_weights,1,size(x)[2]).*x))+x,1)

def fun(x):
    # x is a matrix
    output = np.mean(abs(-np.cos(2*np.pi * x))+x,1)
    return output
#fun = x -> inp_weights'*(abs(-cos(2*π *x))+x)
#fun = x -> sinc(x)
#fun = x -> sin(x)+2

def fun_noise(x, epsilon):
    output = fun(x) -epsilon + 2 *epsilon *np.random.rand(1, x.shape[0])
    return output

global outp_tex_set 
global outp_test_set

outp_tex_set = np.transpose(fun_noise(inp_tex_set, epsilon))
outp_test_set = np.transpose(fun_noise(inp_test_set, epsilon))


global s_tex 
s_tex = inp_tex_set
global f_tex 
f_tex = outp_tex_set

# s_test = inp_dat_perm[:,size_train+1:end]
# f_test = outp_dat_perm[outputdim,size_train+1:end]

#s_test = collect(minimum(inp_dat):.01:maximum(inp_dat))'

# use some data to condition KI on:

cond_dat_size = int(np.floor(s_tex.shape[0] * cond_percentage/100))

global s_tex_cond 
s_tex_cond = s_tex[:,1:cond_dat_size]
global fs_tex_cond 
fs_tex_cond = f_tex[:,1:cond_dat_size]

global D_tex_cond
D_tex_cond = HPA.sample(s_tex_cond,fs_tex_cond,epsilon)

#use some separate data to as test set for training metric hyperparameters:
#partrain_dat_size = size(s_tex,2) - cond_dat_size

global s_tex_partrain 
s_tex_partrain = s_tex[:, cond_dat_size+1:]
global fs_tex_partrain 
fs_tex_partrain = f_tex[:, cond_dat_size+1:]
#s_test_partrain = s_tex
 #fs_test_partrain = f_tex

# Now merge all available data together
global D_tex 
D_tex = HPA.sample(s_tex,f_tex,epsilon)
global s_test 
s_test = inp_test_set

print("Number of tex: " + str(tex_set_size))
print("Test set size: " + str(test_set_size))

print("Input space dim.: " + str(d))

print("mean fvalues (test set): " + str(np.mean(outp_test_set,axis=0))) # TODO: check whether correct axis
print("std fvalues  (test set): " + str(np.std(outp_test_set,0))) # TODO: check whether correct axis


if d == 1:
  s_test = s_test[s_test[:,0].argsort()] # sort


global  f_test 
f_test = np.transpose(fun_noise(s_test, epsilon))
global  x 
x = s_test
global  fx 
fx = fun(x)

# ============= End: data generation =====

print("Beginning training and prediction...")

T_pred = []
T_train = []
mean_err = []
mean_test_err = []
median_test_err = []
median_err = []
std_err = []


# ------------ LACKI -------------------

emptysamp = HPA.sample(np.array([]),np.array([]),0.)
hestthresh_LACKI = 0.
LACKI = HPA.HoelParEst(L, hoelexp, emptysamp, HPA.vecmetric_maxnorm, HPA.vecmetric_2norm,
                       hestthresh_LACKI, par_init_guess, np.array([1.]))
print("Training LACKI...")

tictoc.tic()

HPA.KI_append_sample_N_update_L_(LACKI, s_tex, f_tex, 0.)
T_train_LACKI = tictoc.toc()

print("LACKI: Hoelder constant found:" + str(LACKI.L))

print("Using the LACKI to predict...")

tictoc.tic()

predf_LACKI, temp_err_LACKI, floorpred_LACKI, ceilpred_LACKI  =  HPA.KI_predict(LACKI, s_test) # TODO: check what is "err_LACKI", why assigned twice
predf_LACKI = predf_LACKI # TODO: check whether transpose needed
# err_LACKI = err_LACKI
global T_pred_LACKI 
T_pred_LACKI = tictoc.toc()


global err_LACKI 
err_LACKI = abs(fx - predf_LACKI) # TODO: check whether transpose needed
global test_err_LACKI 
test_err_LACKI = abs(f_test - predf_LACKI)

global mean_err_LACKI 
mean_err_LACKI  = np.mean(err_LACKI)
global mean_test_err_LACKI 
mean_test_err_LACKI = np.mean(test_err_LACKI)

global std_err_LACKI 
std_err_LACKI = np.std(err_LACKI)
global std_test_err_LACKI 
std_test_err_LACKI = np.std(test_err_LACKI)
global median_test_err_LACKI 
median_test_err_LACKI = np.median(test_err_LACKI)
global median_err_LACKI 
median_err_LACKI = np.median(err_LACKI)
#-------------------------------

median_test_err = np.array([median_test_err, median_test_err_LACKI])
median_err = np.array([median_err, median_err_LACKI])
std_err = np.array([std_err, std_err_LACKI])
T_pred = np.array([T_pred, T_pred_LACKI])
T_train = np.array([T_train, T_train_LACKI])
mean_err = np.array([mean_err, mean_err_LACKI])
mean_test_err = np.array([mean_test_err, mean_test_err_LACKI])


# ------------ LACKI2 with noise-------------------
hestthresh_LACKI2 = epsilon
emptysamp = HPA.sample(np.array([]),np.array([]),0.)
LACKI2 = HPA.HoelParEst(L, hoelexp, emptysamp, HPA.vecmetric_maxnorm, HPA.vecmetric_2norm,
                        hestthresh_LACKI2, par_init_guess, np.array([1.]))
print("Training LACKI2...")

tictoc.tic()
HPA.KI_append_sample_N_update_L_(LACKI2,s_tex,f_tex,epsilon)
T_train_LACKI2 = tictoc.toc()



print("LACKI2: Hoelder constant found:" + str(LACKI2.L))

print("Using the LACKI2 to predict...")
tictoc.tic()

predf_LACKI2, temp_err_LACKI2, floorpred_LACKI2, ceilpred_LACKI2 =  HPA.KI_predict(LACKI2,s_test) # TODO: check what is "err_LACKI", why assigned twice
predf_LACKI2 = predf_LACKI2 # TODO: check whether transpose needed
# err_LACKI2 = err_LACKI2 # TODO: check whether transpose needed
global T_pred_LACKI2 
T_pred_LACKI2 = tictoc.toc()

global err_LACKI2 
err_LACKI2 = abs(fx-predf_LACKI2)
global test_err_LACKI2 
test_err_LACKI2 = abs(f_test-predf_LACKI2)

global mean_err_LACKI2 
mean_err_LACKI2 = np.mean(err_LACKI2)
global mean_test_err_LACKI2 
mean_test_err_LACKI2 = np.mean(test_err_LACKI2)

global std_err_LACKI2 
std_err_LACKI2 = np.std(err_LACKI2)
global std_test_err_LACKI2
std_test_err_LACKI2  = np.std(test_err_LACKI2)
global median_test_err_LACKI2 
median_test_err_LACKI2 = np.median(test_err_LACKI2)
global median_err_LACKI2 
median_err_LACKI2 = np.median(err_LACKI2)
#-------------------------------


median_test_err = np.array([median_test_err, median_test_err_LACKI2])
median_err = np.array([median_err, median_err_LACKI2])
std_err = np.array([std_err, std_err_LACKI2])
T_pred = np.array([T_pred, T_pred_LACKI2])
T_train = np.array([T_train, T_train_LACKI2])
mean_err = np.array([mean_err, mean_err_LACKI2])
mean_test_err = np.array([mean_test_err, mean_test_err_LACKI2])

# ===== plotte ====
"""
x = s_test
x = x # TODO check whether need traspose
fx = fx
predf_LACKI = predf_LACKI
predf_LACKI2 = predf_LACKI2

# predf_gpopt = predf_gpopt'
#   predf_gpopt_nonoise = predf_gpoptnonoise'
# predf_POKI_Optimjl = predf_POKI_Optimjl'
# predf_linmod = predf_linmod'
s_tex = s_tex
f_tex = f_tex
f_test = f_test



plt.axis("tight")

plt.title("(2)")

plt.subplot(1,2,1)
plt.figure(figsize=(20,10))
plt.plot(np.transpose(x),f_test,"-", color="cyan",alpha =.2, label = 'noisy test function')
plt.plot(np.transpose(x)[0], predf_LACKI,"-",color="black",alpha =.4,linewidth=2, label = 'prediction')
plt.plot(np.transpose(x)[0],fx,"--", color="darkblue",alpha =.99,linewidth=2, label = 'ground truth')


plt.plot(np.transpose(x)[0],floorpred_LACKI,"-.", color="silver",alpha=.9,linewidth=1, label = 'floor')
plt.plot(np.transpose(x)[0],ceilpred_LACKI,":", color="silver",alpha=.9,linewidth=1, label = 'ceiling')
plt.plot(np.transpose(s_tex),f_tex,".", color="blue",alpha=.9, label = 'f_tex?') #TODO what is f_tex
plt.title("LACKI 1")
# plt.legend(loc=2)



plt.subplot(1,2,2)

plt.plot(np.transpose(x),f_test,"-", color="cyan",alpha =.2)
plt.plot(np.transpose(x)[0],predf_LACKI2, color="black",alpha =.4,linewidth=2)
plt.plot(np.transpose(x)[0],fx,"--", color="darkblue",alpha =.9,linewidth =2)

plt.plot(np.transpose(s_tex),f_tex,".", color="blue",alpha=.9)
plt.plot(np.transpose(x)[0],floorpred_LACKI2,"-.", color="silver",alpha=.9,linewidth=2)
plt.plot(np.transpose(x)[0],ceilpred_LACKI2,":", color="silver",alpha=.9,linewidth=2)
plt.title("LACKI 2")
"""





"""
Created on Tue Sep  4 16:26:21 2018

@author: jianhong
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import HMIN
# x: Rows are samples, samples are vectors

def vecmetric_maxnorm(x, y, pars = [1]):
    # TODO: overloading case where only one sample
    return np.amax(abs(x-y), axis = 1)

def vecmetric_maxnorm_ARD(x, y, pars):
    # x: Rows are samples, samples are vectors
    return np.amax(pars*(abs(x-y)), axis = 1)

def vecmetric_2norm(x, y, pars):
    output = np.sqrt(np.sum((x-y)**2, axis = 1))
    return output

def vecmetric_maxnorm_scaled(x, y, pars):
    a = np.amax(abs(x-y), axis = 1)
    return pars*a

def vecmetric_2norm_scaled(x, y, pars = [1]):
    output = pars[0]*np.sqrt(np.sum((x-y)**2, axis = 1))
    return output

# TODO finish all overloaded functions
"""
def vecmetric_maxnorm_scaled(x, y, pars = [1]):
    output = pars[0] * np.amax(abs(x-y))
    return output
"""

class sample():
    def __init__(self, inp, outp, errbnd = 0):
        self.inp = inp
        self.outp = outp
        self.errbnd = errbnd

    
# To handle different forms of sample, including default errbnd

# TODO different forms of sample

def get_sample_inp(sample):
    return sample.inp

def get_sample_outp(sample):
    return sample.outp

def get_sample_errbnd(sample):
    return sample.errbnd

    
def rem_sample_points_(sample, inds):
    # Removes one sample point from sample matrix based on inds given
    # inds must be list
    ss = sample.inp.shape[0]    
    for ind in inds:
        indkeep = [i for i in range(ss)].pop(ind)        
    sample.inp = sample.inp[indkeep] # keep samples which are in indkeep
    sample.outp = sample.outp[indkeep]    
    return None

def sort1dsample_(sample):
    # Sorts according to inputs, assuming inputs are 1D
    a = sample.inp
    b = a[a[:,0].argsort()] # 0 is hard-coded since this is 1D
    sample.inp = b
    return None

def sortsample_bydimension_(sample, dim):
    # dim is integer
    a = sample.inp
    b = a[a[:,dim].argsort()]
    sample.inp = b
    return None
    
# TODO Julia code doesnt work
def plot_sample(sample):
    return None

def append_sample_(sample, x, f, e):
    # x is array of array
    # f is array of array
    
    if sample.inp == False:
        sample.inp = x
        sample.outp = f
        sample.errbnd = e
        
    else:
        if (sample.inp.size == 0):
            sample.inp = np.array(x)
        else:
            sample.inp = np.vstack((sample.inp, x))
            
        if (sample.outp.size == 0):
            sample.outp = np.array(f)
        else:
            sample.outp = np.vstack((sample.outp, f))
        """
        
        
        """
        sample.errbnd = e



##### HoelParEst 

class HoelParEst():
    def __init__(self, L = 0., p = 1., D = sample(np.array([[0,0]]), np.array([[0,0]]), 0), 
                 fct_metric_inp = vecmetric_2norm, fct_metric_outp = vecmetric_2norm, 
                 hestthresh = 0., pars_metric_inp = np.array([1]), pars_metric_outp = np.array([1])):
        self.L= L
        self.p = p
        self.D = D
        self.fct_metric_inp = fct_metric_inp
        self.fct_metric_outp = fct_metric_outp
        self.hestthresh = hestthresh 
        self.pars_metric_inp = pars_metric_inp
        self.pars_metric_outp = pars_metric_outp
    
    
# TODO : different forms for append_sample_Hoel_
def append_sample_Hoel_(hpa, sample):
    # hpa is type HoelParEst
    append_sample_(hpa.D, sample.inp, sample.outp, sample.errbnd)
    return None

def append_sample_N_update_L_gen_(hpa, x, f, e):
  #slow but general in that it
  # can deal with arbitrary metrics that do not apply to matrices of col vecs
  # however when metrics are defined this way (such as vecmetric_... ) .. often
  #best to use append_sample_N_update_L!
   #update L:
    if np.count_nonzero(hpa.D.inp) == 0 : # if hpa.D.inp is empty
        Lcross = 0
    else:
        Lcross = emp_cross_L_between_samples(hpa.D, x, f, hpa.fct_metric_inp, hpa.fct_metric_outp, hpa.p, hpa.alpha)
    
    l = x.shape[1]
    
    if l <= 1:
        Lnew = 0
    else:
        Lnew = emp_L2(x, f, hpa.fct_metric_inp, hpa.fct_metric_outp, hpa.p, hpa.alpha, 
                      hpa.pars_metric_inp, hpa.pars_metric_outp)
    
    hpa.L = max(hpa.L, Lcross, Lnew)
    
    append_sample_(hpa.D, x, f,e)
    return None

##### Kinky Inference

def KI_predict_N_plot_tight(hpa, x):
    pred, err = KI_predict(hpa, x)

    plt.plot(x, pred, color = [.4, .4, .4], linewidth = 2.0)
    plt.plot(x, pred-err, color = [.7, .7, .7], linewidth = 2.0, linestyle="--",bbox_inches="tight" )
    plt.plot(x, pred+err,color=[.8,.8,.8], linewidth=2.0, linestyle="-.",bbox_inches="tight")
    sample_s = hpa.D.inp;
    sample_f = hpa.D.outp;
    plt.plot(sample_s, sample_f, "go", markersize=9.0, linewidth=10.0, bbox_inches="tight")
    plt.legend(["prediction","floor","ceiling","sample"])    
    # plt.savefig('test.svg')
    return pred, err

def KI_predict_N_plot(hpa, x, predictoronlyflag = False):
    pred, err = KI_predict(hpa, x)
    
    plt.plot(x, pred, color=[.4,.4,.4], linewidth=2.0)
    if predictoronlyflag == False:
        plt.plot(x, pred-err, color=[.8,.8,.8], linewidth=2.0, linestyle="--")
        plt.plot(x, pred+err, color=[.7,.7,.7], linewidth=2.0, linestyle="-.")
        sample_s = hpa.D.inp;
        sample_f = hpa.D.outp;
        plt.plot(sample_s,sample_f,"go",markersize=9.0,linewidth=10.0)
        plt.legend(["prediction","floor","ceiling","sample"]) 
    else:
        sample_s = hpa.D.inp;
        sample_f = hpa.D.outp;
        plt.plot(sample_s,sample_f,"go",markersize=9.0,linewidth=10.0)
        plt.legend(["prediction","sample"])

    return pred,err


def KI_predict(hpa, x):
    #x is assumed to be a matrix of col vector inputs
    # x[0] gives number of vectors
    #pred will be a vector of prediction values
    sample_s = hpa.D.inp
    sample_f = hpa.D.outp
    epsilon = hpa.D.errbnd
    n = x.shape[0] # number of test inputs
    ns = len(sample_f)
    ceilpred = np.ones(n)
    floorpred = np.ones(n)
    pred = np.ones(n)
    err = math.inf
    
    if np.count_nonzero(sample_s) == 0:
        return None
    else:
        # go through test input by test input
        for iter in range(n):
            X = np.array([x[iter] for col in range(ns)]) # now ith col fo x stacked next to itself for each tex
            m_rowvec = hpa.L * (hpa.fct_metric_inp(X, sample_s, hpa.pars_metric_inp)**hpa.p) # all the distances of inp in one row vec            
            sample_f = np.squeeze(sample_f)

            floorpred[iter] = np.amax(sample_f - epsilon - m_rowvec)
            ceilpred[iter] = np.amin(sample_f + epsilon + m_rowvec)
            pred[iter] = (ceilpred[iter]+floorpred[iter])/2
        err = (ceilpred - floorpred)/2
    
    return pred, err, floorpred, ceilpred

def Lip_quad_1d_batch(sample, L, I):
    # D is "sample" object
    # I is domain interval in vector
    # L is Lipschitz costant
    # This function performs Lipschitz quadrature
    sort1dsample_(sample)
    Ds = sample.inp
    Df = sample.outp
    De = sample.errbnd
    N = len(Ds)
    if N > 0:
        Df_upper = Df + De
        Df_lower = Df = De
        
        if N == 1:
            Su = Df_upper[0]*( I[1] - I[0]) + (L/2.)*( (I[1]-Ds[0])^2+(Ds[0]-I[0] )**2)
            Sl = Df_lower[0]*( I[1] - I[0]) - (L/2.)*( (I[1]-Ds[0])^2+(Ds[0]-I[0] )**2)
        elif N>1:
            # computation of breakpoints
            xi = np.zeros(N-1)
            for i in range(N-1):
                xi[i] = 0.5*(Ds[i] + Ds[i+1] + (Df_upper[i+1] - Df_upper[i])/L)
                
            xi = np.concatenate(np.array(I[0]), xi, np.array(I[1]))
            
            Su = 0
            Sl = 0
            
            for i in range(N):
                Su = Su + Df_upper[i]*( xi[i+1] - xi[i]) + (L/2)*( (xi[i+1]-Ds[i])^2+(Ds[i]-xi[i] )**2)
                Sl = Sl + Df_lower[i]*( xi[i+1] - xi[i]) - (L/2)*( (xi[i+1]-Ds[i])^2+(Ds[i]-xi[i] )**2)
    
        est_S = (Su + Sl)/2 # integral estimate
        est_bnd = (Su - Sl)/2 # error bound around integral estimate
    
    else:
        est_S = 0
        est_bnd = math.inf
        Su = math.inf
        Sl = -math.inf
        
    return est_S, est_bnd, Su, Sl

def KI_reset_(hpa, hpa_new = HoelParEst()):
    hpa = copy.deepcopy(hpa_new)
    return None

# Main difference to KI_append_sample_N_update_L! is that Lest is updated first but then
#test pt only included if under the new estimated L, the new data is within the prediction error

def KI_append_sample_N_update_L_v2_(hpa, x, f, e):
    # f is array of 1-val-array
    # x is array of vector arrays
    n = x.shape[0] # number of samples
    
    # now see if we need to update Hoelder const L
    if np.count_nonzero(hpa.D.inp) == 0:
        append_sample_(hpa.D, x[0], f[0], e)
        
        if n > 1:
            KI_append_sample_N_update_L_(hpa, x[1:], f[1:], e)
        
        return None
    
    else:
        #go through test input by test input:
        #compute empirical const est:
        for i in range(n):
            outp = hpa.D.outp
            ns = outp.shape[0] # ns = number of samples in old data
            X = np.array([x[i] for col in range(ns)])
            m_rowvec = hpa.fct_metric_inp(X, hpa.D.inp, hpa.pars_metric_inp)
            inds = m_rowvec > 0
            mr = m_rowvec[inds]
            
            if np.count_nonzero(mr) == 0:
                F = np.array([f[i] for col in range(ns)])
                diffs_f = hpa.fct_metric_outp(F, hpa.D.outp, hpa.pars_metric_outp) - hpa.alpha
                hpa.L = max(hpa.L, max(diffs_f[inds]/(mr**hpa.p)))
                predn, prederrn = KI_predict(hpa, x[i]) # TODO find out what '' does
                if abs(predn[0] - f[i][0]) <= prederrn[0]:
                    append_sample_(hpa.D, x[i], f[i][0], e)
        
        return None
"""
OLD

def KI_append_sample_N_update_L_(hpa, x, f, e):
    n = x.shape[0] # number of samples
    
    if np.count_nonzero(hpa.D.inp) == 0: # is hpa.D.inp empty
        #append_sample_(hpa.D, x[0], f[0], e)
        append_sample_(hpa.D, x, f, e)
        if n > 1:
            KI_append_sample_N_update_L_(hpa, x, f, e)
                  
        return None

    else:
        
        sample_s_old = hpa.D.inp
        sample_f_old = hpa.D.outp

        #go through test input by test input:
        #compute empirical const est:
        for i in range(n):
            #now ith col of x stacked on next to itself for each
            #tex:
            ns = sample_f_old.shape[0] # ns = number of samples in old data
            X = np.array([x[i] for col in range(ns)])
            m_rowvec=hpa.fct_metric_inp(X,sample_s_old,hpa.pars_metric_inp)
            inds = m_rowvec > 0
            #print("ns")
            #print(ns)
            mr = m_rowvec[inds]
            print('m_rowvec')
        
            if np.count_nonzero(mr) != 0:
                F = np.array([f[i] for col in range(ns)])
                diffs_f = hpa.fct_metric_outp(F, hpa.D.outp, hpa.pars_metric_outp) - hpa.alpha
                hpa.L = max(hpa.L, max(diffs_f[inds]/(mr**hpa.p)))
                print('updating L')
                print(sample_s_old)
                print(x[i])
                #sample_s_old = np.concatenate([sample_s_old, x[i]], axis = 0)
                #sample_f_old = np.concatenate([sample_f_old, f[i]], axis = 0)
                sample_s_old = np.vstack((sample_f_old, f[i]))
                sample_f_old = np.vstack((sample_f_old, f[i]))
    # append_sample_(hpa.D, x, f, e)

    return None
"""


def KI_append_sample_N_update_L_(hpa, x, f, e):
    n = x.shape[0] # number of samples
    
    if np.count_nonzero(hpa.D.inp) == 0: # is hpa.D.inp empty
        append_sample_(hpa.D, x[0], f[0], e)
        if n > 1:
            KI_append_sample_N_update_L_(hpa, x[1:], f[1:], e)
                  
        return None

    else:
        
        sample_s_old = hpa.D.inp
        sample_f_old = hpa.D.outp
        
        #go through test input by test input:
        #compute empirical const est:
        for i in range(n):
            #now ith col of x stacked on next to itself for each
            #tex:
            ns = sample_f_old.shape[0] # ns = number of samples in old data
            X = np.array([x[i] for col in range(ns)])
            m_rowvec = hpa.fct_metric_inp(X,sample_s_old,hpa.pars_metric_inp)
            inds = m_rowvec != 0
            
            mr = m_rowvec[inds]

            if np.count_nonzero(mr) != 0:
                F = np.array([f[i] for col in range(ns)])
                diffs_f = hpa.fct_metric_outp(F, sample_f_old, hpa.pars_metric_outp) - hpa.hestthresh

                hpa.L = max(hpa.L, max(diffs_f[inds]/(mr**hpa.p)))

                #print("L " + str(hpa.L))

                sample_s_old = np.vstack((sample_s_old, x[i]))
                sample_f_old = np.vstack((sample_f_old, f[i]))
    append_sample_(hpa.D, x, f, e)

    return None


def KI_predict_Real2Real(hpa, t):
    predf = KI_predict(hpa,t)[0]
    return predf[0]


def KI_append_sample_N_update_LNp_(hpa, x, f, e):
    #x: row vec of col vecs
    L = hpa.L
    p = hpa.p
    sample_f_old =hpa.D.outp
    sample_s_old =hpa.D.inp; #contains the old sample
    n = x.shape[0]
    
    if np.count_nonzero(sample_s_old) == 0:
        append_sample_(hpa.D, x, f, e)
    
        if n > 1:
            KI_append_sample_N_update_L_(hpa, x[1:], f[1:], e)
    
    else:
        for i in range(n):
            ns = sample_f_old.shape[0] # ns = number of samples in old data
            X = np.array([x[i] for col in range(ns)])
            dx_rowvec = hpa.fct_metric_inp(X, sample_s_old, hpa.pars_metric_inp)
            F = np.array([f[i] for col in range(ns)])
            df_rowvec = hpa.fct_metric_outp(F, sample_f_old, hpa.pars_metric_outp)- hpa.alpha
            inds = dx_rowvec > 0
            if L > 0: #filter out those positions that are valid condidates for Hoelder exponent updates
                inds2 = inds & (df_rowvec < L) & (dx_rowvec < 1.)
                df2 = df_rowvec[inds2]
                
                if len(df2) > 0:
                    r = math.log(df2/L) / math.log(dx_rowvec[inds2])
                    r = min(r)
                    p = min(p,r) 
                  # r = log(dx_rowvec[inds2])./log(df_rowvec[inds2]./L)
                  # w = maximum(r)
                  # #println(length(r))
                  # p = min(p,1/w) #Hoelder constant updated
                      #Hoelder constant updated
                        #print("Updater says: found p = $r")
                df2 = df_rowvec[~inds2]
                if len(df2) > 0:
                    L = max(L, max(df2/(dx_rowvec[~inds2]**p)))
                else:
                    L = max(L, max(df_rowvec[inds]/(dx_rowvec[inds]**p)))
                
                sample_s_old = np.concatenate([sample_s_old, x[i]], axis = 0)
                sample_s_old = np.concatenate([sample_f_old, f[i]], axis = 0)
            
    hpa.p = p
    hpa.L = L
    append_sample_(hpa.D, x, f, e)
    return None


##### KI-Metric function parameter optimisation routines

def KI_fct_loss_pars_metric_inp_testdat(obj, normpar, testdata4paropt_s, testdata4paropt_f):
    """
    loss for optimizing parameter of norm (here frequency) -- based on separate test data fit
    
    Assumes function output is real-valued
    N = testdata4paropt_s.shape[0]
    
    obj = HoelParEst 
    """
    pars_metric_inp_backup = obj.pars_metric_inp
    obj.pars_metric_inp = normpar
    loss = abs(KI_predict(obj, testdata4paropt_s)[0] - testdata4paropt_f) # mean across all samples for each individual dimension
    mean_loss = np.mean(loss, axis=1)
    obj.pars_metric_inp = pars_metric_inp_backup
    return mean_loss

def plot_KI_fct_loss_pars_metric_inp_testdat(obj, parmin, parmax, testdata4paropt_s, testdata4paropt_f):
    #assumes a 1-dim inp space of parameters
    def fun(p):
        return KI_fct_loss_pars_metric_inp_testdat(obj, p, testdata4paropt_s, testdata4paropt_f)
    
    x = np.linspace(parmin, parmax, num = 3000)
    fx = np.zeros(3000)
    
    for i in range(x):
        fx[i] = fun(x[i])[0]
    plt.plot(x, fx)
    return None
    
## TODO plot_KI has another form


def KI_optimise_pars_metric_inp_testdat_VS_(obj, parmin, parmax, testdata4paropt_s = np.array([]), 
                                            testdata4partopt_f = np.array([]), L = -999.123, 
                                            maxevals = 10000, errthresh = 0.05):
    # optimise metric par on separate test data
    if np.count_nonzero(testdata4paropt_s) == 0:
        testdata4paropt_s = obj.D.inp
        testdata4paropt_f = obj.D.outp
        
    if L == -999.123:
        L = obj.L
    
    def fct(par):
        return KI_fct_loss_pars_metric_inp_testdat(obj,par,testdata4paropt_s,testdata4paropt_f)
    # TODO: this package!
    optobj = HMIN_VS.HMINVSTOR(parmin, parmax, fct, L)
    # opobj - HMIN.HMIN_multidim(parmin, parmax, fct, L)
    
    argmin, m, i, counter = HMIN_VS.minimiseUntilErrthresh_(optobj, errthresh, maxevals)
    #theta = fminbnd(@obj.fct_normparloss,thetamin,thetamax);
    #theta = fminunc(@obj.fct_normparloss,abs(thetamax-thetamin)/2);
    obj.pars_metric_inp = argmin
    print("KI_optimise: number of iterations for parameter optimisation: " + str(counter))
    
    return None


def KI_optimise_pars_metric_inp_testdat_SHATTER_(obj, parmin, parmax, testdata4paropt_s = np.array([]), 
                                                 testdata4partopt_f = np.array([]), L = -999.123,
                                                maxevals = 10000, errthresh = 0.05):
    if np.count_nonzero(testdata4paropt_s) == 0:
        testdata4paropt_s = obj.D.inp
        testdata4paropt_f =obj.D.outp
    
    if L == -999.123:
        L = obj.L
    
    def fct(par):
        return KI_fct_loss_pars_metric_inp_testdat(obj, par, testdata4paropt_s,testdata4paropt_f)[0]
    # TODO: this package!
    optobj = HMIN_VS_SHATTER.HMIN_SHATTER(parmin, parmax, fct, L)
    # optobj = HMIN.HMIN_multidim(parmin,parmax,fct,L)
    argmin,m,i,counter = HMIN_VS_SHATTER.minimiseUntilErrthresh_(optobj,errthresh,maxevals)
    obj.pars_metric_inp = argmin
    print("KI_optimise: number of iterations for parameter optimisation: "+ str(counter))
    return None


def KI_optimise_pars_metric_inp_testdat_(obj, parmin, parmax, testdata4paropt_s = np.array([]), 
                                                 testdata4partopt_f = np.array([]), L = -999.123,
                                                maxevals = 10000, errthresh = 0.05):
    if np.count_nonzero(testdata4paropt_s) == 0:
        testdata4paropt_s = obj.D.inp
        testdata4paropt_f =obj.D.outp
    
    if L == -999.123:
        L = obj.L

    def fct(par):
        return KI_fct_loss_pars_metric_inp_testdat(obj, par, testdata4paropt_s,testdata4paropt_f)

    optobj = HMIN.HMIN_multidim(parmin,parmax,fct,L)
    argmin,m,i,counter = HMIN.minimiseUntilErrthresh_(optobj,errthresh,maxevals)
    
    obj.pars_metric_inp = argmin
    print("KI_optimise: number of iterations for parameter optimisation: " + str(counter))
    return None

# TODO not sure about packages
"""
def KI_optimise_pars_metric_inp_testdat_Optimjl_(obj, parmin, parmax, testdata4paropt_s = np.array([]),
                                                 testdata4paropt_f= np.array([]), maxevals = 10000):
    if np.count_nonzero(testdata4paropt_s) == 0:
        testdata4paropt_s = obj.D.inp
        testdata4paropt_f =obj.D.outp
    
    parinit = 0.5*parmin + 0.5*parmax
    
    if len(parinit) == 1:
        def fct(par):
            return KI_fct_loss_pars_metric_inp_testdat(obj, np.array(par), testdata4paropt_s,testdata4paropt_f)[0]
        res = Optim.optimize(fct, parmin[0], parmax[0])
        # TODO which package
    else:
        def fct(par):
            return KI_fct_loss_pars_metric_inp_testdat(obj, par, testdata4paropt_s,testdata4paropt_f)[0]
        res = Optim.optimize(fct, parmin[0], parmax[0])
    argmin = np.array(Optim.minimizer(res)) # argmin = vec(collect(Optim.minimizer(res)))
    m = Optim.minimum(res) 
    obj.pars_metric_inp = argmin
    print("KI_optimise: number of iterations for parameter optimisation: " + str(Optim.iterations(res))) 
"""

def KI_optimise_pars_metric_inp_testdat_Shubert_(obj, parmin, parmax,testdata4paropt_s= np.array([]), 
                                                 testdata4paropt_f = np.array([]),L=100.,
                                                 maxevals = 10000, errthresh = 0.05):
    if np.count_nonzero(testdata4paropt_s) == 0:
        testdata4paropt_s = obj.D.inp
        testdata4paropt_f = obj.D.outp
    
    if len(parmin) == 1:
        def fct(par):
            return KI_fct_loss_pars_metric_inp_testdat(obj, par, testdata4paropt_s, testdata4paropt_f)[0]
        
        argmin,minval,numevals = HMIN.minimise_Shubert(fct,[parmin[0],parmax[0]],L,errthresh,maxevals)
    else:
        print("Shubert's method can only be employed for 1-dimensional functions!")
    
    obj.pars_metric_inp = argmin
    print("KI_optimise_pars_metric_inp_testdat_Shubert!: number of iterations for parameter optimisation: " + str(numevals))
    return None

def KI_optimise_pars_metric_inp_testdat_old_(obj, parmin, parmax, testdata4paropt_s= np.array([]),
                                         testdata4paropt_f=np.array([]),L=-999.123,maxevals=10e4):
    if np.count_nonzero(testdata4paropt_s) == 0:
        testdata4paropt_s = obj.D.inp
        testdata4paropt_f = obj.D.outp
    
    if L == -999.123:
        L = obj.L
        
    def fct(par):
        return KI_fct_loss_pars_metric_inp_testdat(obj,par,testdata4paropt_s,testdata4paropt_f)
    
    optobj = HMIN.HMIN_multidim(parmin, parmax, fct, L)
    argmin,m,i,counter = HMIN.minimiseUntilErrthresh_old_(optobj,.05,maxevals)
    obj.pars_metric_inp = argmin
    print("KI_optimise: number of iterations for parameter optimisation: " + str(counter))
    return None

##### Utility functions

def emp_cross_L_between_samples2(x1, f1, x2, f2, fct_metric_inp, fct_metric_outp, p=1., alpha= 0,
                                 pars_metric_inp = np.array([1.]), pars_metric_outp = np.array([1.])):
    ns1 = x1.shape[0]
    ns2 = x2.shape[0]
    
    if ns1 <= 1 & ns2 <=1:
        return 0
    
    L = 0
    for j in range(ns2):
        X = np.array([x2[j] for col in range(ns1)])
        F = np.array([f2[j] for col in range(ns1)])
        m_rowvec = fct_metric_inp(X, x1, pars_metric_inp)
        inds = m_rowvec > 0
        m_rowvec = m_rowvec[inds]**p
        diffs_f = fct_metric_outp(F, f1, pars_metric_outp) - alpha
        diffs_f = diffs_f[inds]
        L = max(L, max(diffs_f/m_rowvec))
    
    return L

def emp_cross_L_between_samples(x1, f1, x2, f2, fct_metric_inp, fct_metric_outp, p=1.,
                                alpha=0., pars_metric_inp = np.array([1.]), pars_metric_outp = np.array([1.])):
    ns1 = x1.shape[0]
    ns2 = x2.shape[0]
    
    if ns1 <= 1 & ns2 <=1:
        return 0

    L = 0
    
    for i in range(ns1):
        for j in range(ns2):
            dx = fct_metric_inp(x1[i], x2[j], pars_metric_inp)
            if dx > 0.:
                L = max(L, (fct_metric_outp(f1[i], f2[j], pars_metric_outp) - alpha)/(dx**p)) # Lip const estimator
    
    return L

# TODO: variosu forms for emp_cross_L_between_samples

def emp_L(x, f, fct_metric_inp, fct_metric_outp, p = 1., alpha = 0., pars_metric_inp = np.array([1.]), 
          pars_metric_outp = np.array([1.])):
    ns = x.shape[0]
    if ns <= 1:
        return 0
    L = 0
    
    for i in range(ns):
        for j in range(i-1):
            dx = fct_metric_inp(x[i], x[j], pars_metric_inp)
            if dx > 0:
                L = max(L, (fct_metric_outp(f[i], f[j], pars_metric_outp) - alpha)/(dx**p))
    
    return L


def emp_L2(x, f, fct_metric_inp, fct_metric_outp, p = 1., alpha = 0., pars_metric_inp = np.array([1.]), 
          pars_metric_outp = np.array([1.])):
    ns = x.shape[0]
    if ns <= 1:
        return 0
    L = 0
    
    for j in range(ns):
        if j+1 < ns:
            xcmp = x[j+1:]
            fcmp = f[j+1:]
            nxcmp = xcmp.shape[0]
            X = np.array([x[j] for col in range(nxcmp)])
            F = np.array([f[j] for col in range(nxcmp)])
            m_rowvec = fct_metric_inp(X, xcmp, pars_metric_inp)
            inds = m_rowvec > 0
            m_rowvec = m_rowvec[inds]**p
            diffs_f = fct_metric_outp(F, fcmp, pars_metric_outp) - alpha
            diffs_f = diffs_f[inds]
            L = max(L, max(diffs_f/m_rowvec))
    return L


                
##### Bayesian belief update over L based on Pareto density

def density_pareto(x, pars):
    # m = min parameter > 0
    # nu = shape parameter (nu = 0: uninformative)
    m = pars[0]
    nu = pars[1]
    p = np.zeros(len(x))
    for i in range(len(x)):
        xi = x[i]
        if xi >= m:
            p[i] = nu * (m**nu) / (xi**(nu+1))
    
    return p
    

class HoelParEst_Bayes():
    def __init__(self, pars_density, p = 1., D = sample(np.array([]), np.array([]), 0.), 
                 fct_metric_inp = vecmetric_2norm, fct_metric_outp = vecmetric_2norm, 
                 alpha = 0., pars_metric_inp = np.array([1.]), pars_metric_outp = np.array([1.])):
        self.pars_density = pars_density
        self.p = p
        self.D = D
        self.fct_metric_inp = fct_metric_inp
        self.fct_metric_outp = fct_metric_outp
        self.alpha = alpha
        self.pars_metric_inp = pars_metric_inp
        self.pars_metric_outp = pars_metric_outp


def append_sample_N_update_L_belief_pareto_(hpa, x, f, e): 
    # == probably want to use this over append_sample_N_update_L_gen!
    if np.count_nonzero(hpa.D.inp) == 0:
        Lcross = 0
    else:
        Lcross = emp_cross_L_between_samples2(hpa.D.inp,hpa.D.outp,x,f,hpa.fct_metric_inp,hpa.fct_metric_outp,
                                              hpa.p,hpa.alpha,hpa.pars_metric_inp,hpa.pars_metric_outp)
    l = x.shape[0]
    if l <= 1:
        Lnew = 0
    else:
        Lnew = emp_L2(x, f, hpa.fct_metric_inp, hpa.fct_metric_outp,
                      hpa.p, hpa.alpha, hpa.pars_metric_inp, hpa.pars_metric_outp)
    #update 1st pareto parameter:
    hpa.pars_density[0] = max(hpa.pars_density[0],Lcross,Lnew)
    #update 2nd pareto parameter:
    hpa.pars_density[1] += l #add the number of tex to the shape parameter
    #append the new data points:
    append_sample_(hpa.D, x, f, e)    
    return None
    
    
    
    
        
        
        
        
        
    
    
    
    
    
    
        
        
        
     
    
        
    
    
    
    
    
        

                    

                    
                
            
    

    
    
    

    
    
    
    
    
    
    

    
            
            
        
    
        
    
    

        
    
              
            
            
            
    
    
    
    
    
        

    
    
        
    
    
           
        



    



    










#testing Kinky inference with ARD metric with weights of dimensions
#optimised to min the neg log likelihood of the test data (under Laplace distr) / ie the average error
#on the test data.
import PyPlot
import HPA

# ----- Begin: Parameters of the test run -----------
f = x -> sin(abs(x[1,:] -.5))+1 #function defined so that only the first input dimension matters
L=1.
cond_dat_size =5
partrain_dat_size =5
epsilon =0. #obs noise
hoelexp = 1.
d = 4 #dimensionality of the input space
par_init_guess = ones(d)
maxevals = 20000 #max. number of evaluations in parameter training
parmin = 0.001 * ones(d) # guess of minimal parameter (weight of dim)
parmax = 1.* ones(d) # guess of maximal parameter for each inp dim

# ----- End: Parameters of the test run -----------

# ----- Begin: data generation -----------

x = rand(d,1000)
fx = f(x)


#use some data to condition KI on:
s_cond = x[:,randperm(size(x,2))[1:cond_dat_size]]#pick a few random sample points
fs_cond = f(s_cond)

D_cond= HPA.sample(s_cond,fs_cond,epsilon)

#use some separate data to as test set for training metric hyperparameters:
s_test_partrain = x[:,randperm(size(x,2))[1:partrain_dat_size]]#pick a few random sample points
fs_test_partrain = f(s_test_partrain)

# Now merge all available data together
D_cond_full = HPA.sample([s_cond s_test_partrain],[fs_cond fs_test_partrain],epsilon)
# ----- End: data generation -----------

# ------------ creation of the KI learner based on D_cond_full and false frequency guess:


ki0 = HPA.HoelParEst(L,hoelexp,D_cond_full,HPA.vecmetric_maxnorm_ARD,HPA.vecmetric_2norm,0.,par_init_guess,collect(1.))

# ------------ creation of the KI learner only based on D_cond, but with trained and false frequency guess:
ki1_VS = HPA.HoelParEst(L,hoelexp,D_cond,HPA.vecmetric_maxnorm_ARD,HPA.vecmetric_2norm,0.,par_init_guess,collect(1.))
Lparopt = ki1_VS.L


@time HPA.KI_optimise_pars_metric_inp_testdat_VS!(ki1_VS,parmin,parmax,s_test_partrain,fs_test_partrain,Lparopt,maxevals)
println("ki1_VS: Parameter found:",ki1_VS.pars_metric_inp )
ki1_VS.D = D_cond_full






ki1 = HPA.HoelParEst(L,hoelexp,D_cond,HPA.vecmetric_maxnorm_ARD,HPA.vecmetric_2norm,0.,par_init_guess,collect(1.))
Lparopt = ki1.L


@time HPA.KI_optimise_pars_metric_inp_testdat!(ki1,parmin,parmax,s_test_partrain,fs_test_partrain,Lparopt,maxevals)
println("ki1: Parameter found:",ki1.pars_metric_inp )
ki1.D = D_cond_full





## ---------------- Plot results:
#PyPlot.figure(1)
#PyPlot.plot(x,fx)
predf0 =  HPA.KI_predict(ki0,x)[1]'
err0 = abs(fx-predf0)
#PyPlot.figure(2)
#PyPlot.plot(x,fx)
predf1 =  HPA.KI_predict(ki1,x)[1]'
#PyPlot.show()
err1 = abs(fx-predf1)

PyPlot.boxplot((err0,err1))

show()


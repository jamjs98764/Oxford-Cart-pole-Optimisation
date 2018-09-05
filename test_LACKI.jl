#testing Kinky inference with metric_scaled with scale_parameter coinciding with the
#Hoelder const under noisy data
#using Debug
#this time, we replace the KIARD method by a GP
import GaussianProcesses
#import Loess
import PyPlot
#import linregression
import HPA
close("all")
# -------- Parameters:





outputdim =1 #set to 1 .. can be set to different vals if data has multiple output dimensions
L=1. #should be set to 1
epsilon =1.#assumed obs noise

hoelexp = .7
#d = 1 #dimensionality of the input space
par_init_guess = collect(1.)
maxevals = 4000#max. number of evaluations in parameter training
Lmin = 0.
Lmax = 10.




d =1 #input space dimensionality

inp_weights = 1./collect(1:d)

test_set_size = 1500
#tex_percentage = 2 #percentage of sample used for training
cond_percentage = 50

numlearners =2
numtestruns =1


# ---- let's go....
T_pred_hist =  zeros(numtestruns,numlearners)
T_train_hist =  zeros(numtestruns,numlearners)
mean_err_hist =  zeros(numtestruns,numlearners)
std_err_hist =  zeros(numtestruns,numlearners)
mean_test_err_hist =  zeros(numtestruns,numlearners)
median_test_err_hist =  zeros(numtestruns,numlearners)
median_err_hist = zeros(numtestruns,numlearners)



  tex_set_size  = 200
#   texpercentage = 4+randperm(15)[1]

println("\n")

# ================ DATA EXTRACTION AND PREPARTION ================
global inp_test_set = 1*rand(d,test_set_size)
global inp_tex_set = rand(d,tex_set_size)

#PyPkgfun = x -> minimum(repmat(inp_weights,1,size(x)[2]).*(abs(-cos(2*π *x))+x),1)
#fun = x -> mean(repmat(inp_weights,1,size(x)[2]).*(abs(-cos(2*π *x))+x),1)
#fun = x ->(abs(-cos(2*π *x)))[1,:]
#fun = x ->mean(abs(-cos(2*π *HPA.vecmetric_2norm(0*x,repmat(inp_weights,1,size(x)[2]).*x))),1)
#fun = x -> mean(abs(-cos(2*π *x))+x,1)
#fun = x -> mean(abs(-cos(2*π *repmat(inp_weights,1,size(x)[2]).*x))+x,1)
fun = x -> mean(abs(-cos(2*π * x))+x,1)
#fun = x -> inp_weights'*(abs(-cos(2*π *x))+x)
#fun = x -> sinc(x)
#fun = x -> sin(x)+2

fun_noise = x -> fun(x) +-.5 *epsilon + rand(Float64,(1,size(x)[2]))
#fun_noise = x -> fun(x) + .25randn((1,size(x)[2]))

global outp_tex_set = fun_noise(inp_tex_set)
global outp_test_set = fun_noise(inp_test_set)
#inp_dat = readcsv("../../data/cart2pend/inp.dat")
#outp_dat = readcsv("../../data/cart2pend/outp.dat")


global s_tex = inp_tex_set
global f_tex = outp_tex_set

# s_test = inp_dat_perm[:,size_train+1:end]
# f_test = outp_dat_perm[outputdim,size_train+1:end]

#s_test = collect(minimum(inp_dat):.01:maximum(inp_dat))'


#use some data to condition KI on:
cond_dat_size =convert(Int64,floor(size(s_tex,2)*cond_percentage/100))


global s_tex_cond = s_tex[:,1:cond_dat_size]
global fs_tex_cond = f_tex[:,1:cond_dat_size]

global D_tex_cond= HPA.sample(s_tex_cond,fs_tex_cond,epsilon)

#use some separate data to as test set for training metric hyperparameters:
#partrain_dat_size = size(s_tex,2) - cond_dat_size

global s_tex_partrain = s_tex[:,cond_dat_size+1:end]
global fs_tex_partrain = f_tex[:,cond_dat_size+1:end]
#s_test_partrain = s_tex
 #fs_test_partrain = f_tex

# Now merge all available data together
global D_tex = HPA.sample(s_tex,f_tex,epsilon)
global s_test = inp_test_set




println("Number of tex: ",tex_set_size)
println("Test set size:", test_set_size)

println("Input space dim.: ",d)

println("mean fvalues (test set): ",mean(outp_test_set,2))
println("std fvalues  (test set): ",std(outp_test_set,2))




if d ==1
  s_test = sort(collect(s_test))'
end

global  f_test = fun_noise(s_test)
global  x = s_test
global  fx = fun(x)
# ============= End: data generation =====



println("Beginning training and prediction...")

  T_pred = []
  T_train = []
  mean_err = []
  mean_test_err = []
 median_test_err = []
  median_err = []
  std_err = []




# ------------ LACKI -------------------

emptysamp = HPA.sample(zeros(0,0),zeros(0,0),.0)
  hestthresh_LACKI = 0.
LACKI = HPA.HoelParEst(L,hoelexp,emptysamp,HPA.vecmetric_maxnorm,HPA.vecmetric_2norm,hestthresh_LACKI,par_init_guess,collect(1.))
println("Training LACKI...")

tic()
HPA.KI_append_sample_N_update_L!(LACKI,s_tex,f_tex,0.)
T_train_LACKI =toc()



println("LACKI: Hoelder constant found:",LACKI.L )

println("Using the LACKI to predict...")
tic()
predf_LACKI,err_LACKI,floorpred_LACKI,ceilpred_LACKI  =  HPA.KI_predict(LACKI,s_test)
predf_LACKI = predf_LACKI'
err_LACKI = err_LACKI'
global T_pred_LACKI =toc()

global err_LACKI = abs(fx-predf_LACKI)'
global test_err_LACKI = abs(f_test-predf_LACKI)'

global mean_err_LACKI  = mean(err_LACKI)
global mean_test_err_LACKI  = mean(test_err_LACKI)

global std_err_LACKI  = std(err_LACKI)
global std_test_err_LACKI  = std(test_err_LACKI)
global median_test_err_LACKI = median(test_err_LACKI)
  global median_err_LACKI = median(err_LACKI)
#-------------------------------
  median_test_err = [median_test_err; median_test_err_LACKI]
  median_err = [median_err; median_err_LACKI]
    std_err = [std_err; std_err_LACKI]
T_pred = [T_pred; T_pred_LACKI]
  T_train = [T_train; T_train_LACKI]
  mean_err = [mean_err; mean_err_LACKI]

  mean_test_err = [mean_test_err; mean_test_err_LACKI]


# ------------ LACKI2 with noise-------------------
hestthresh_LACKI2 = epsilon
emptysamp = HPA.sample(zeros(0,0),zeros(0,0),.0)
LACKI2 = HPA.HoelParEst(L,hoelexp,emptysamp,HPA.vecmetric_maxnorm,HPA.vecmetric_2norm,hestthresh_LACKI2,par_init_guess,collect(1.))
println("Training LACKI2...")

tic()
HPA.KI_append_sample_N_update_L!(LACKI2,s_tex,f_tex,epsilon)
T_train_LACKI2 =toc()



println("LACKI2: Hoelder constant found:",LACKI2.L )

println("Using the LACKI2 to predict...")
tic()
 predf_LACKI2,err_LACKI2,floorpred_LACKI2,ceilpred_LACKI2 =  HPA.KI_predict(LACKI2,s_test)
predf_LACKI2 = predf_LACKI2'
err_LACKI2 = err_LACKI2'
global T_pred_LACKI2 =toc()

global err_LACKI2 = abs(fx-predf_LACKI2)'
global test_err_LACKI2 = abs(f_test-predf_LACKI2)'

global mean_err_LACKI2  = mean(err_LACKI2)
global mean_test_err_LACKI2  = mean(test_err_LACKI2)

global std_err_LACKI2  = std(err_LACKI2)
global std_test_err_LACKI2  = std(test_err_LACKI2)
global median_test_err_LACKI2 = median(test_err_LACKI2)
  global median_err_LACKI2 = median(err_LACKI2)
#-------------------------------
  median_test_err = [median_test_err; median_test_err_LACKI2]
  median_err = [median_err; median_err_LACKI2]
    std_err = [std_err; std_err_LACKI2]
T_pred = [T_pred; T_pred_LACKI2]
  T_train = [T_train; T_train_LACKI2]
  mean_err = [mean_err; mean_err_LACKI2]

  mean_test_err = [mean_test_err; mean_test_err_LACKI]

# ===== plotte ====

  x=s_test
x = x'
fx = fx'
predf_LACKI = predf_LACKI'
predf_LACKI2 = predf_LACKI2'
# predf_gpopt = predf_gpopt'
#   predf_gpopt_nonoise = predf_gpoptnonoise'
# predf_POKI_Optimjl = predf_POKI_Optimjl'
# predf_linmod = predf_linmod'
s_tex = s_tex'
f_tex = f_tex'
f_test = f_test'



PyPlot.figure(1)
  PyPlot.axis("tight")

PyPlot.title("(2)")

PyPlot.subplot(121)
PyPlot.plot(x,f_test,"-", color="cyan",alpha =.2)
PyPlot.plot(x,predf_LACKI,"-",color="black",alpha =.4,linewidth=2)
PyPlot.plot(x,fx,"--", color="darkblue",alpha =.99,linewidth=2)


PyPlot.plot(x,floorpred_LACKI,"-.", color="silver",alpha=.9,linewidth=1)
PyPlot.plot(x,ceilpred_LACKI,":", color="silver",alpha=.9,linewidth=1)
PyPlot.plot(s_tex,f_tex,".", color="blue",alpha=.9)
PyPlot.title("LACKI 1")
PyPlot.legend(["noisy test function","prediction","ground truth","floor","ceiling"],loc=2)




PyPlot.subplot(122)
PyPlot.plot(x,f_test,"-", color="cyan",alpha =.2)
PyPlot.plot(x,predf_LACKI2, color="black",alpha =.4,linewidth=2)
PyPlot.plot(x,fx,"--", color="darkblue",alpha =.9,linewidth =2)
PyPlot.plot(s_tex,f_tex,".", color="blue",alpha=.9)
PyPlot.plot(x,floorpred_LACKI2,"-.", color="silver",alpha=.9,linewidth=2)
PyPlot.plot(x,ceilpred_LACKI2,":", color="silver",alpha=.9,linewidth=2)
PyPlot.title("LACKI 2")

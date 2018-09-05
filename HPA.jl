module HPA #Hoelder parameter adaptation
# general convention: vectors are col vectors, data of vectors are matrices where vectors are the columns. so sequences are indexed by columns
#push!(LOAD_PATH, "/Users/jpc73/Documents/RESEARCH/code/juliacode/mymodules")
include("../mymodules/HMIN_VS.jl")
include("../mymodules/HMIN_VS_SHATTER.jl")
include("../mymodules/HMIN.jl")
importall Base
import Optim
import PyPlot
export HoelParEst, sample, KI_predict, KI_predict_N_plot, append_sample!

   #--------------------metrics-----------------:
  # --------------- assume they compute metrics between x[:,i] and y[:,i] for i =1...
  # result is a row vector of distances

vecmetric_maxnorm(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1}) =  maximum(abs(x-y),1)#treats matrices as collections of vectors
vecmetric_maxnorm(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1}) = maximum(abs(x-y))
vecmetric_maxnorm(x,y) =vecmetric_maxnorm(x,y,[1.])

#@debug
function vecmetric_maxnorm_ARD(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1})
#@bp
maximum(repmat(pars,1,size(x,2)).* abs(x-y),1)
#treats matrices as collections of vectors
end


function vecmetric_maxnorm_scaled(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1})
#@bp
repmat(pars,1,size(x,2)).*maximum(abs(x-y),1)
#treats matrices as collections of vectors
end
vecmetric_2norm_scaled(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1}) = pars[1]*sqrt(sumabs2(x-y,1))
vecmetric_2norm_scaled(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1}) = pars[1]*sqrt(sumabs2(x-y))
vecmetric_2norm_scaled(x,y) =vecmetric_2norm_scaled(x,y,[1.])

vecmetric_maxnorm_scaled(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1}) = pars[1]*maximum(abs(x-y))
vecmetric_maxnorm_scaled(x,y) =vecmetric_maxnorm_scaled(x,y,[1.])

vecmetric_maxnorm_ARD(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1}) = maximum(pars.*abs(x-y))
vecmetric_maxnorm_ARD(x,y) =vecmetric_maxnorm(x,y,[1.])

vecmetric_2norm(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1}) = sqrt(sumabs2(x-y,1))
vecmetric_2norm(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1}) = sqrt(sumabs2(x-y))
vecmetric_2norm(x,y) = vecmetric_2norm(x,y,[1.])

vecmetric_periodic(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1}) = abs(sin(pars[1].*pi*sqrt(sumabs2(x-y,1))))
vecmetric_periodic(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1}) = abs(sin(pars[1].*pi.*sqrt(sumabs2(x-y))))
vecmetric_periodic(x,y) = vecmetric_periodic(x,y,[1.])

 # ================= best not to use these (although they are generic) :
metric_maxnorm(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1})= norm(x-y,Inf)
metric_maxnorm(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1}) = norm(x-y,Inf)#treats matrices via matrix norm
metric_maxnorm(x,y) = metric_maxnorm(x,y,[1.])

metric_2norm(x::Array{Float64,1},y::Array{Float64,1},pars::Array{Float64,1}) = norm(x-y,2)
metric_2norm(x::Array{Float64,2},y::Array{Float64,2},pars::Array{Float64,1}) = norm(x-y,2)
metric_2norm(x,y) = metric_2norm(x,y,[1.])


#    ============ Sample types and methods on them

# ----------------- sample type 1



type sample
   inp::Array{Float64,2}  #x
   outp::Array{Float64,2} #  \tilde f(x) where tilde f(x) \in [f(x)-e,f(x) +e]
   errbnd::Float64
 end
  sample(inp::Array{Float64,2},outp::Array{Float64,2}) = sample(inp,outp,0.)
  sample(inp::Array{Float64,1},outp::Array{Float64,1},e) = sample(reshape(inp,1,length(inp)),reshape(outp,1,length(outp)),e)
  sample(inp::Array{Float64,1},outp::Array{Float64,1}) = sample(reshape(inp,1,length(inp)),reshape(outp,1,length(outp)),0.)
  sample(inp::Array{Float64,1},outp::Float64,e) = sample(reshape(inp,length(inp),1),reshape([outp],1,1),e)
  sample(inp::Array{Float64,1},outp::Float64) = sample(reshape(inp,length(inp),1),reshape([outp],1,1),0.)
  sample(inp::Array{Float64,2},outp::Array{Float64,1},err::Float64) = sample_err(inp,outp,err)


get_sample_inp(D::sample) = return D.inp
get_sample_outp(D::sample) = return D.outp
get_sample_errbnd(D::sample) = return D.errbnd


function rem_sample_pts!(s::sample,inds::Vector{Int64})
  ss = size(s.inp,2)
  indkeep = setdiff(collect(1:ss),inds)
  s.inp = s.inp[:,indkeep]
  s.outp = s.outp[:,indkeep]
end

function sort1dsample!(D::sample) #sorts according to inputs
  if ~issorted(D.inp) #sort the grid
               Dmat=sortrows([D.inp' D.outp']);
               D.inp = [Dmat[:,1]]'
               D.outp = [Dmat[:,2]]'
  end
end

function plot_sample(D::sample)
  if D.errbnd > Inf # DOES NOT WORK YET, HENCE DISABLED
    e = ones(1,length(D.outp))
    p= PyPlot.errorbar(D.inp,D.outp,yerr=[D.errbnd .* e;D.errbnd.*e],fmt="o")
  else
    p=PyPlot.scatter(D.inp,D.outp,s=16,alpha=.5,bbox_inches="tight")
  end
end

function append_sample!(D::sample,x::Array{Float64,2},f::Array{Float64,2},e::Float64)
  if isempty(D.inp)
    D.inp = x
    D.outp =f
    D.errbnd = e
  else
    D.inp = [D.inp x]
    D.outp = [D.outp f]
    D.errbnd = max(D.errbnd, e)
  end
end

append_sample!(D::sample,x::Array{Float64,2},f::Array{Float64,2})= append_sample!(D::sample,x::Array{Float64,2},f::Array{Float64,2},0.)
append_sample!(D::sample,D2::sample) = append_sample!(D::sample,D2.inp::Array{Float64,2},D2.outp::Array{Float64,2},D2.errbnd::Float64)
#append_sample!(D::Array{None,1},x::Array{Float64,2},f::Array{Float64,2},e::Float64) = D=sample(x,f,e)
#append_sample!(D::Array{None,1},x::Array{Float64,2},f::Array{Float64,2}) = D=sample(x,f)
append_sample!(D::sample,x::Array{Float64,1},f::Array{Float64,1},e)= append_sample!(D::sample,x',f',e)
append_sample!(D::sample,x::Array{Float64,1},f::Float64,e)= append_sample!(D::sample,x',collect(f)',e)
append_sample!(D::sample,x::Float64,f::Float64,e)= append_sample!(D::sample,collect(x)',collect(f)',e)
append_sample!(D::sample,x::Matrix{Float64},f::Float64,e::Float64)=append_sample!(D,x,collect(f)',e)

function append_sample!(D::sample,x::Array{Float64,2},f::Matrix{Float64},e::Float64)
  if isempty(D.inp)
    D.inp = x
    D.outp =f
    D.errbnd = e
  else
    D.inp = [D.inp x]
    D.outp = [D.outp f]
    D.errbnd = max(D.errbnd, e)
  end
end




# =================== HoelParEst type ================


type HoelParEst
  #container for current paramter
   L::Float64#Hoelder constant est
   p::Float64#Hoelder exponent est
   D::sample  #Data of the form {(s,f(s),e)}
   fct_metric_inp::Function
   fct_metric_outp::Function
   alpha::Float64 #threshold for Lip const est
   pars_metric_inp::Vector{Float64}
   pars_metric_outp::Vector{Float64}
end


  #type HoelParEst_Bayes <: HoelParEst
  #  pars_belief_density::Vector()
  #end
HoelParEst(L,p,D,f1,f2,alpha) = HoelParEst(L,p,D,f1,f2,alpha,[1.],[1.])
HoelParEst(L,p,D,f1,f2) = HoelParEst(L,p,D,f1,f2,0.)
HoelParEst(L,p,D,f1) = HoelParEst(L,p,D,f1,vecmetric_2norm)
HoelParEst(L,p,D) = HoelParEst(L,p,D,vecmetric_2norm,vecmetric_2norm,0.)
HoelParEst(L,p) = HoelParEst(L,p,sample(zeros(0,0),zeros(0,0),0.),vecmetric_2norm,vecmetric_2norm,0.)
HoelParEst(L) = HoelParEst(L,1.)
HoelParEst() = HoelParEst(0.,1.)

append_sample!(h::HoelParEst,D2::sample) = append_sample!(h.D2,D2.inp,D2.outp,D2.errbnd)
append_sample!(h::HoelParEst,x::Float64,f::Float64,e::Float64)=append_sample!(h.D,collect(x)',collect(f)',0.)
append_sample!(h::HoelParEst,x::Float64,f::Float64)=append_sample!(h.D,collect(x)',collect(f)',0.)
append_sample!(h::HoelParEst,x::Vector{Float64},f::Vector{Float64})=append_sample!(h.D,x',f',0.)
append_sample!(h::HoelParEst,x::Vector{Float64},f::Vector{Float64},e::Float64)=append_sample!(h.D,x',f',e)
append_sample!(h::HoelParEst,x::Vector{Float64},f::Float64,e::Float64)=append_sample!(h.D,x',collect(f)',e)
append_sample!(h::HoelParEst,x::Matrix{Float64},f::Float64,e::Float64)=append_sample!(h.D,x,collect(f)',e)

function append_sample_N_update_L_gen!(hpa::HoelParEst,x::Array{Float64,2},f::Array{Float64,2},e::Float64)#slow but general in that it
  # can deal with arbitrary metrics that do not apply to matrices of col vecs
  # however when metrics are defined this way (such as vecmetric_... ) .. often
  #best to use append_sample_N_update_L!
   #update L:

   if isempty(hpa.D.inp)
     Lcross = 0
   else
    Lcross=emp_cross_L_between_samples(hpa.D,x,f,hpa.fct_metric_inp,hpa.fct_metric_outp,hpa.p,hpa.alpha)
   end
  l =size(x,2)
  if l <= 1
    Lnew =0
  else
   Lnew =emp_L(x,f::Matrix{Float64},hpa.fct_metric_inp::Function,hpa.fct_metric_outp::Function,hpa.p::Float64,hpa.alpha)
  end
    hpa.L =maximum([hpa.L;Lcross;Lnew])
  #append the new data points:
   append_sample!(hpa.D,x,f,e)
 end

function append_sample_N_update_L!(hpa::HoelParEst,x::Array{Float64,2},f::Array{Float64,2},e::Float64) # == probably want to use this over append_sample_N_update_L_gen!
   #update L:

   if isempty(hpa.D.inp)
     Lcross = 0
   else
    Lcross=emp_cross_L_between_samples2(hpa.D,x,f,hpa.fct_metric_inp,hpa.fct_metric_outp,hpa.p,hpa.alpha,hpa.pars_metric_inp,hpa.pars_metric_outp)
   end
  if size(x,2) <= 1
    Lnew =0
  else
    Lnew =emp_L2(x,f,hpa.fct_metric_inp,hpa.fct_metric_outp,hpa.p,hpa.alpha,hpa.pars_metric_inp,hpa.pars_metric_outp)
  end
    hpa.L =maximum([hpa.L;Lcross;Lnew])
  #append the new data points:
   append_sample!(hpa.D,x,f,e)
 end




# ========== Kinky Inference ============================
function KI_predict_N_plot_tight(hpa::HoelParEst,x::Matrix{Float64})
  (pred,err) =KI_predict(hpa,x);
  PyPlot.svg(true)
               PyPlot.plot(x',pred,color=[.1,.1,.1], linewidth=2.0,bbox_inches="tight")
                PyPlot.plot(x',pred-err,color=[.7,.7,.7], linewidth=2.0, linestyle="--",bbox_inches="tight")
                PyPlot.plot(x',pred+err,color=[.8,.8,.8], linewidth=2.0, linestyle="-.",bbox_inches="tight")
               sample_s = hpa.D.inp;
                sample_f = hpa.D.outp;
                 PyPlot.plot(sample_s,sample_f,"og",markersize=9.0,linewidth=10.0,bbox_inches="tight")
  PyPlot.legend(["prediction","floor","ceiling","sample"])
  return pred,err
end



function KI_predict_N_plot(hpa::HoelParEst,x::Matrix{Float64},predictoronlyflag =false)
  (pred,err) =KI_predict(hpa,x);
  PyPlot.svg(true)
               PyPlot.plot(x',pred,color=[.4,.4,.4], linewidth=2.0)
                if predictoronlyflag == false
                  PyPlot.plot(x',pred-err,color=[.8,.8,.8], linewidth=2.0, linestyle="--")
                  PyPlot.plot(x',pred+err,color=[.7,.7,.7], linewidth=2.0, linestyle="-.")
               sample_s = hpa.D.inp;
                sample_f = hpa.D.outp;
                 PyPlot.plot(sample_s,sample_f,"og",markersize=9.0,linewidth=10.0)
 # PyPlot.legend(["prediction","floor","ceiling","sample"])
  else
               sample_s = hpa.D.inp;
                sample_f = hpa.D.outp;
                 PyPlot.plot(sample_s,sample_f,"og",markersize=9.0,linewidth=10.0)
        #         PyPlot.legend(["prediction","sample"])
  end
  return pred,err
end

        KI_predict(hpa::HoelParEst,x::Float64) = KI_predict(hpa,collect(x)')
        KI_predict(hpa::HoelParEst,x::Vector{Float64}) = KI_predict(hpa,x'')

function KI_predict(hpa::HoelParEst,x::Matrix{Float64})
            #x is assumed to be a matrix of col vector inputs
            #pred will be a vector of prediction values
                sample_s = hpa.D.inp;
                sample_f = hpa.D.outp;
                epsilon = hpa.D.errbnd;
            n = size(x,2); #number of test inputs
            ns = length(sample_f);
            ceilpred = ones(n);
            floorpred = ones(n);
            pred =ones(n);
            err =Inf;
            if isempty(sample_s)
                return;
            else
                #go through test input by test input:
                for i =1:n
                    #now ith col of x stacked next to itself for each
                    #tex:
                    X =repmat(x[:,i],1,ns)
                    #take abs-differences:
            m_rowvec=hpa.L .* hpa.fct_metric_inp(X,sample_s,hpa.pars_metric_inp).^(hpa.p) #all the distances of inp in one row vec
                    floorpred[i] = maximum(sample_f -epsilon  - m_rowvec);
                    ceilpred[i]  = minimum(sample_f +epsilon+ m_rowvec);
                    pred[i] =(ceilpred[i]+floorpred[i])/2;
                end
    ## the following block was in to enforce bounds
#                 if obj.minval_fct(x) > -Inf
#                     m =obj.minval_fct(x)'.*ones(1,n);
#                     floorpred=max([floorpred';m])';
#                     ceilpred=max([ceilpred';m])';
#                     pred =(ceilpred+floorpred)/2;
#                 end
#                 if obj.maxval_fct(x) < Inf
#                     m =obj.maxval_fct(x)'.*ones(1,n);
#                     floorpred=min([floorpred';m])';
#                     ceilpred=min([ceilpred';m])';
#                     pred =(ceilpred+floorpred)/2;
#                 end
                err = (ceilpred-floorpred)/2;
            end
          return pred,err,floorpred,ceilpred
  #!!!!!!!! returns a column vector!
        end

function Lip_quad_1d_batch(D::sample,L::Float64,I = Vector{Float64})
  #Lipschhitzquadrature
  #I: domain interval
  #L: Lipschitz constant
              sort1dsample!(D)
                Ds= D.inp
                Df=D.outp
                De = D.errbnd
                N = length(Ds);
            if N >0
                Df_upper = Df+De; Df_lower = Df-De;


                if N == 1
                    Su = Df_upper[1]*( I[2] - I[1]) + (L/2.)*( (I[2]-Ds[1])^2+(Ds[1]-I[1] )^2);
                    Sl = Df_lower[1]*( I[2] - I[1]) - (L/2.)*( (I[2]-Ds[1])^2+(Ds[1]-I[1] )^2);
                elseif N >1
                       #computation of breakpoints:
                     xi = zeros(N-1);
                            for i=1:N-1
                                xi[i] = .5*( Ds[i] + Ds[i+1] + (Df_upper[i+1] - Df_upper[i])./L );
                            end

                            xi = [I[1]; xi; I[2]];
                            Su =0; Sl=0;
                            for i=1:N
                                Su = Su + Df_upper[i]*( xi[i+1] - xi[i]) + (L/2)*( (xi[i+1]-Ds[i])^2+(Ds[i]-xi[i] )^2);
                                Sl = Sl + Df_lower[i]*( xi[i+1] - xi[i]) - (L/2)*( (xi[i+1]-Ds[i])^2+(Ds[i]-xi[i] )^2);
                            end

                end

                est_S = (Su+Sl)./2;#integral estimate
                est_bnd = (Su-Sl)./2; #error bound around the integral estimate
            else
                est_S = 0;
                est_bnd = Inf;
                Su = Inf;
                Sl = -Inf;
            end
  return est_S,est_bnd,Su,Sl
end



function KI_reset!(hpa::HoelParEst,hpa_new::HoelParEst=HoelParEst())
  hpa = deepcopy(hpa_new)
end


# Main difference to KI_append_sample_N_update_L! is that Lest is updated first but then
#test pt only included if under the new estimated L, the new data is within the prediction error
function KI_append_sample_N_update_L_v2!(hpa::HoelParEst,x::Array{Float64,2},f::Matrix{Float64},e::Float64)
            #f: row vec
            #x: row vec of col vecs
            n = size(x,2);
            #now see if we need to update Hoelder const L:
            if isempty(hpa.D.inp)
                append_sample!(hpa.D,x[:,1],f[:,1],e);
                if n>1
                    KI_append_sample_N_update_L!(hpa,x[:,2:end],f[:,2:end],e)
                end
                return;
            else

               #sample_s_old =hpa.D.inp#contains the old sample + oddly if we dont preallocate its faster...
               #sample_f_old = hpa.D.outp

                #go through test input by test input:
                #compute empirical const est:
                for i =1:n
                    #now ith col of x stacked on next to itself for each
                    #tex:
                    ns = size(hpa.D.outp,2)#ns = number of samples in old data
                    X =repmat(x[:,i],1,ns)
                    m_rowvec=hpa.fct_metric_inp(X,hpa.D.inp,hpa.pars_metric_inp)
                    inds = m_rowvec .> 0
                    mr = m_rowvec[inds]
                    if ~isempty(mr)
                        F =repmat(f[:,i],1,ns)
                        diffs_f = hpa.fct_metric_outp(F,hpa.D.outp,hpa.pars_metric_outp) - hpa.alpha
                        hpa.L =maximum([hpa.L;maximum(diffs_f[inds]./(mr.^(hpa.p)))])
                        #sample_s_old =  [sample_s_old x[:,i]]#oddly if we dont preallocate its faster...
                        #sample_f_old= [sample_f_old f[:,i]]
                        predn,prederrn=KI_predict(hpa,x[:,i]'')#returns a vector of values (with one element)
                        if abs(predn[1] - f[1,i]) <= prederrn[1]
                          append_sample!(hpa.D,x[:,i]'',f[1,i]'',e)
                        end

                    end
                end

            end
           # append_sample!(hpa.D,x,f,e)

 end
 function KI_append_sample_N_update_L!(hpa::HoelParEst,x::Array{Float64,2},f::Matrix{Float64},e::Float64)
            #f: row vec
            #x: row vec of col vecs
            n = size(x,2);
            #now see if we need to update Hoelder const L:
            if isempty(hpa.D.inp)
                append_sample!(hpa.D,x[:,1]'',f[:,1]'',e);
                if n>1
                    KI_append_sample_N_update_L!(hpa,x[:,2:end],f[:,2:end],e)
                end
                return;
            else

               sample_s_old =hpa.D.inp#contains the old sample + oddly if we dont preallocate its faster...
               sample_f_old = hpa.D.outp

                #go through test input by test input:
                #compute empirical const est:
                for i =1:n
                    #now ith col of x stacked on next to itself for each
                    #tex:
                    ns = size(sample_f_old,2)#ns = number of samples in old data
                    X =repmat(x[:,i],1,ns)
                    m_rowvec=hpa.fct_metric_inp(X,sample_s_old,hpa.pars_metric_inp)
                    inds = m_rowvec .> 0
                    mr = m_rowvec[inds]
                    if ~isempty(mr)
                        F =repmat(f[:,i],1,ns)
                        diffs_f = hpa.fct_metric_outp(F,sample_f_old,hpa.pars_metric_outp) - hpa.alpha
                        hpa.L =maximum([hpa.L;maximum(diffs_f[inds]./(mr.^(hpa.p)))])
                        sample_s_old =  [sample_s_old x[:,i]]#oddly if we dont preallocate its faster...
                        sample_f_old= [sample_f_old f[:,i]]


                    end
                end

            end
            append_sample!(hpa.D,x,f,e)

 end

function KI_predict_Real2Real(hpa::HoelParEst,t::Float64)
  predf = HPA.KI_predict(hpa,t)[1]
  return predf[1]
end

function KI_append_sample_N_update_LNp!(hpa::HoelParEst,x::Array{Float64,2},f::Matrix{Float64},e::Float64)
            #f: row vec
            #x: row vec of col vecs
            L = hpa.L
            p = hpa.p
            sample_f_old =hpa.D.outp
            sample_s_old =hpa.D.inp; #contains the old sample
            n = size(x,2);
            #now see if we need to update Hoelder const L:
            if isempty(sample_s_old)
                append_sample!(hpa.D,x[:,1],f[:,1],e);
                if n>1
                    KI_append_sample_N_update_LNp!(hpa,x[:,2:end],f[:,2:end],e)
                end
                return;
            else



                #go through test input by test input:
                #compute empirical const est:
                for i =1:n
                    #now ith col of x stacked on next to itself for each
                    #tex:
                    ns = size(sample_f_old,2);
                    X =repmat(x[:,i],1,ns);
                    dx_rowvec=hpa.fct_metric_inp(X,sample_s_old,hpa.pars_metric_inp) #
                    F =repmat(f[:,i],1,ns);
                    df_rowvec = hpa.fct_metric_outp(F,sample_f_old,hpa.pars_metric_outp) - hpa.alpha
                    inds = dx_rowvec .> 0
                    if L >0
                      #filter out those positions that are valid condidates for Hoelder exponent updates:
                      inds2 = inds & (df_rowvec .< L ) & (dx_rowvec .< 1. )
                      df2 =df_rowvec[inds2]
                      if length(df2) >0
  #                       r = log(dx_rowvec[inds2])./log(df_rowvec[inds2]./L)
  #                       w = maximum(r)
  #                       #println(length(r))
  #                       p = min(p,1/w) #Hoelder constant updated
                        r = log(df2./L)./log(dx_rowvec[inds2])
                        r = minimum(r)
                        p = min(p,r) #Hoelder constant updated
                        #print("Updater says: found p = $r")

                      end
                      #for the remaining entries we do Hoelder const updates (could use ~inds2 but doubtful this will save computation)
                      #so perhaps over inds is best?
                      df2 =df_rowvec[~inds2]
                      if length(df2) >0
                        L =maximum([L;maximum(df2./(dx_rowvec[~inds2].^(p)))]);
                      end
                    else # just update Lipschitz constant
                      L =maximum([L;maximum(df_rowvec[inds]./(dx_rowvec[inds].^(p)))]);
                    end

                    sample_s_old = [sample_s_old x[:,i]];
                    sample_f_old = [sample_f_old f[:,i]];
                end

            end
  hpa.p = p
  hpa.L = L
  append_sample!(hpa.D,x,f,e)
 end
# ------------------- BEGIN: KI-Metric function parameter optimisation routines


function KI_fct_loss_pars_metric_inp_testdat(obj::HoelParEst,normpar,testdata4paropt_s::Matrix{Float64},testdata4paropt_f::Matrix{Float64})#loss for optimizing parameter of norm (here frequency) -- based on separate test data fit
            #assumes function output is real-valued
            #N = size(testdata4paropt_f,2)
            pars_metric_inp_backup = obj.pars_metric_inp
            obj.pars_metric_inp = normpar
            loss = mean(abs(KI_predict(obj,testdata4paropt_s)[1]' - testdata4paropt_f),2)#assumes function output is real-valued
            obj.pars_metric_inp = pars_metric_inp_backup
            return loss
          end

function plot_KI_fct_loss_pars_metric_inp_testdat(obj::HoelParEst,parmin::Float64,parmax::Float64,testdata4paropt_s::Matrix{Float64},testdata4paropt_f::Matrix{Float64})
#assumes a 1-dim inp space of parameters.
  fun = p -> KI_fct_loss_pars_metric_inp_testdat(obj::HoelParEst,p,testdata4paropt_s::Matrix{Float64},testdata4paropt_f::Matrix{Float64})
  x = vec(linspace(parmin,parmax,3000))
  fx = zeros(3000)
  for i=1:length(x)
    fx[i] = fun([x[i]])[1]
  end
  PyPlot.plot(x,fx)
  #PyPlot.axis("tight")
  end
plot_KI_fct_loss_pars_metric_inp_testdat(obj::HoelParEst,parmin::Vector{Float64},parmax::Vector{Float64},testdata4paropt_s::Matrix{Float64},testdata4paropt_f::Matrix{Float64})= plot_KI_fct_loss_pars_metric_inp_testdat(obj::HoelParEst,parmin[1],parmax[1],testdata4paropt_s::Matrix{Float64},testdata4paropt_f::Matrix{Float64})
         function KI_optimise_pars_metric_inp_testdat_VS!(obj::HoelParEst,parmin::Vector{Float64},parmax::Vector{Float64},testdata4paropt_s=[],testdata4paropt_f=[],L::Float64=-999.123,maxevals::Int64=10^5,errthresh =0.05)#optimise metric par on separate test data
            if isempty(testdata4paropt_s)
              testdata4paropt_s = obj.D.inp;
              testdata4paropt_f =obj.D.outp;
            end
            if L == -999.123 #if no L provided as an argument
              L = obj.L
            end
              fct = par -> KI_fct_loss_pars_metric_inp_testdat(obj,par,testdata4paropt_s,testdata4paropt_f)
            optobj = HMIN_VS.HMIN_VSTOR(parmin,parmax,fct,L)
            #optobj = HMIN.HMIN_multidim(parmin,parmax,fct,L)
             argmin,m,i,counter = HMIN_VS.minimiseUntilErrthresh!(optobj,errthresh,maxevals)
            #theta = fminbnd(@obj.fct_normparloss,thetamin,thetamax);
            #theta = fminunc(@obj.fct_normparloss,abs(thetamax-thetamin)/2);
            obj.pars_metric_inp = argmin
            println("KI_optimise: number of iterations for parameter optimisation:",counter)
         end

    function KI_optimise_pars_metric_inp_testdat_SHATTER!(obj::HoelParEst,parmin::Vector{Float64},parmax::Vector{Float64},testdata4paropt_s=[],testdata4paropt_f=[],L::Float64=-999.123,maxevals::Int64=10^5,errthresh =0.05)#optimise metric par on separate test data
            if isempty(testdata4paropt_s)
              testdata4paropt_s = obj.D.inp;
              testdata4paropt_f =obj.D.outp;
            end
            if L == -999.123 #if no L provided as an argument
              L = obj.L
            end
              fct = par -> KI_fct_loss_pars_metric_inp_testdat(obj,par,testdata4paropt_s,testdata4paropt_f)[1]
            optobj = HMIN_VS_SHATTER.HMIN_SHATTER(parmin,parmax,fct,L)
            #optobj = HMIN.HMIN_multidim(parmin,parmax,fct,L)
             argmin,m,i,counter = HMIN_VS_SHATTER.minimiseUntilErrthresh!(optobj,errthresh,maxevals)
            #theta = fminbnd(@obj.fct_normparloss,thetamin,thetamax);
            #theta = fminunc(@obj.fct_normparloss,abs(thetamax-thetamin)/2);
            obj.pars_metric_inp = argmin
            println("KI_optimise: number of iterations for parameter optimisation:",counter)
         end

         function KI_optimise_pars_metric_inp_testdat!(obj::HoelParEst,parmin::Vector{Float64},parmax::Vector{Float64},testdata4paropt_s=[],testdata4paropt_f=[],L::Float64=-999.123,maxevals::Int64=10^5,errthresh=0.05)#optimise metric par on separate test data
            if isempty(testdata4paropt_s)
              testdata4paropt_s = obj.D.inp;
              testdata4paropt_f =obj.D.outp;
            end
            if L == -999.123 #if no L provided as an argument
              L = obj.L
            end
              fct = par -> KI_fct_loss_pars_metric_inp_testdat(obj,par,testdata4paropt_s,testdata4paropt_f)
           # optobj = HMIN.HMIN_VSTOR(parmin,parmax,fct,L)
            optobj = HMIN.HMIN_multidim(parmin,parmax,fct,L)
             argmin,m,i,counter = HMIN.minimiseUntilErrthresh!(optobj,errthresh,maxevals)
            #theta = fminbnd(@obj.fct_normparloss,thetamin,thetamax);
            #theta = fminunc(@obj.fct_normparloss,abs(thetamax-thetamin)/2);
            obj.pars_metric_inp = argmin
             println("KI_optimise: number of iterations for parameter optimisation:",counter)
         end

function KI_optimise_pars_metric_inp_testdat_Optimjl!(obj::HoelParEst,parmin::Vector{Float64},parmax::Vector{Float64},testdata4paropt_s=[],testdata4paropt_f=[],maxevals::Int64=10^5)#optimise metric par on separate test data
            if isempty(testdata4paropt_s)
              testdata4paropt_s = obj.D.inp;
              testdata4paropt_f =obj.D.outp;

            end
              parinit = .5*parmin+ .5 *parmax

                           if length(parinit)==1
                         fct = par -> KI_fct_loss_pars_metric_inp_testdat(obj,vec([par]),testdata4paropt_s,testdata4paropt_f)[1]

                        res= Optim.optimize(fct,parmin[1],parmax[1])

                          else
                         fct = par -> KI_fct_loss_pars_metric_inp_testdat(obj,par,testdata4paropt_s,testdata4paropt_f)[1]
                          res= Optim.optimize(fct,parinit)
                    #res= Optim.optimize(fct,parinit,parmin,parmax,Optim.Fminbox())


            end
          argmin = vec(collect(Optim.minimizer(res)))
          m = Optim.minimum(res)

            obj.pars_metric_inp = argmin
             println("KI_optimise: number of iterations for parameter optimisation:",Optim.iterations(res))
end

# the following only works for 1-dim parameters
 function KI_optimise_pars_metric_inp_testdat_Shubert!(obj::HoelParEst,parmin::Vector{Float64},parmax::Vector{Float64},testdata4paropt_s=[],testdata4paropt_f=[],L::Float64=100.,maxevals::Int64=10^5,errthresh=0.05)
            if isempty(testdata4paropt_s)
              testdata4paropt_s = obj.D.inp;
              testdata4paropt_f =obj.D.outp;

            end

                          if length(parmin)==1
    fct = par -> KI_fct_loss_pars_metric_inp_testdat(obj,vec(collect(par)),testdata4paropt_s,testdata4paropt_f)[1]
                            argmin,minval,numevals = HMIN.minimise_Shubert(fct,[parmin[1];parmax[1]],L,errthresh,maxevals)
                          else
                            error("Shubert's method can only be employed for 1-dimensional functions!")
                          end
            obj.pars_metric_inp = collect(argmin)
            println("KI_optimise_pars_metric_inp_testdat_Shubert!: number of iterations for parameter optimisation:",numevals)
end


         function KI_optimise_pars_metric_inp_testdat!_old(obj::HoelParEst,parmin::Vector{Float64},parmax::Vector{Float64},testdata4paropt_s=[],testdata4paropt_f=[],L::Float64=-999.123,maxevals::Int64=10^5)#optimise metric par on separate test data
            if isempty(testdata4paropt_s)
              testdata4paropt_s = obj.D.inp;
              testdata4paropt_f =obj.D.outp;
            end
            if L == -999.123 #if no L provided as an argument
              L = obj.L
            end
              fct = par -> KI_fct_loss_pars_metric_inp_testdat(obj,par,testdata4paropt_s,testdata4paropt_f)
              #optobj = HMIN.HMIN_VSTOR(parmin,parmax,fct,L)
              optobj = HMIN.HMIN_multidim(parmin,parmax,fct,L)
              argmin,m,i,counter = HMIN.minimiseUntilErrthresh!_old(optobj,.05,maxevals)
            #theta = fminbnd(@obj.fct_normparloss,thetamin,thetamax);
            #theta = fminunc(@obj.fct_normparloss,abs(thetamax-thetamin)/2);
            obj.pars_metric_inp = argmin
             println("KI_optimise: number of iterations for parameter optimisation:",counter)
         end
# ------------------- END: KI- Metric function parameter optimisation routines --------------------

# ================= utility funtions: ===========================




function emp_cross_L_between_samples2(x1::Array{Float64,2},f1::Array{Float64,2},x2::Array{Float64,2},f2::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64,pars_metric_inp::Vector{Float64} =[1.],pars_metric_outp::Vector{Float64}=[1.])
     ns1 = size(x1,2)#number of sample pts
     ns2 = size(x2,2)
     if ns1 <=1 && ns2 <=1
      return 0
     end
     L = 0
     for j=1:ns2
     X = repmat(x2[:,j],1,ns1)
     F = repmat(f2[:,j],1,ns1)
     m_rowvec=fct_metric_inp(X,x1,pars_metric_inp) #
    inds = m_rowvec .> 0
     m_rowvec = (m_rowvec[inds]).^p
     diffs_f = fct_metric_outp(F,f1,pars_metric_outp) - alpha
    diffs_f = diffs_f[inds]
    #diffs_f = abs(sample_f_old-f[i]) - hpa.alpha
     L =max(L,maximum(diffs_f./m_rowvec));

     end
   return L
   end

function emp_cross_L_between_samples(x1::Array{Float64,2},f1::Array{Float64,2},x2::Array{Float64,2},f2::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64,pars_metric_inp::Vector{Float64}=[1.],pars_metric_outp::Vector{Float64}=[1.])
     ns1 = size(x1,2)#number of sample pts
     ns2 = size(x2,2)
     if ns1 <=1 && ns2 <=1
      return 0
     end
     L = 0.
     for i=1:ns1
     for j=1:ns2
      dx = fct_metric_inp(x1[:,i],x2[:,j],pars_metric_inp)
       if dx >0.
         L = max(L,(fct_metric_outp(f1[:,i],f2[:,j],pars_metric_outp)-alpha) ./ (dx.^p)) #Lip const estimator
       end
     end
   end
   return L
   end

emp_cross_L_between_samples(dat::sample,dat2::sample,fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64)= emp_cross_L_between_samples(dat.inp::Array{Float64,2},dat.outp::Array{Float64,2},dat2.inp::Array{Float64,2},dat2.outp::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p::Float64,alpha::Float64)
emp_cross_L_between_samples(dat::sample,x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p::Float64,alpha::Float64)= emp_cross_L_between_samples(dat.inp::Array{Float64,2},dat.outp::Array{Float64,2},x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p::Float64,alpha::Float64)
emp_cross_L_between_samples(x1::Array{Float64,2},f1::Array{Float64,2},x2::Array{Float64,2},f2::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64) =emp_cross_L_between_samples(x1::Array{Float64,2},f1::Array{Float64,2},x2::Array{Float64,2},f2::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64,[1.],[1.])
   #Compute empirical Hoelder const based on a sample
   function emp_L(x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64,pars_metric_inp::Vector{Float64}=[1.],pars_metric_outp::Vector{Float64}=[1.])
     #returns empirical Hoelder const L computed on the basis of sample in dat
     ns = size(x,2)#number of sample pts
     if ns <=1
      return 0
     end
     L = 0.
     for i=1:ns
       for j=1:i-1
        dx = fct_metric_inp(x[:,i],x[:,j],pars_metric_inp)
        if dx >0
          L = max(L,(fct_metric_outp(f[:,i],f[:,j],pars_metric_outp)-alpha) ./ (dx.^p))    #Lip const estimator
        end
       end
     end
   return L
   end
emp_L(x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64)=emp_L(x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64,[1.],[1.])


function emp_L2(x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64,pars_metric_inp::Vector{Float64}=[1.],pars_metric_outp::Vector{Float64}=[1.])
     ns = size(x,2)#number of sample pts

     if ns <=1
      return 0
     end
     L = 0
     for j=1:ns
       if j+1 < ns
         xcmp = x[:,j+1:end]
         fcmp = f[:,j+1:end]
         nxcmp = size(xcmp,2)
         X = repmat(x[:,j],1,nxcmp)
         F = repmat(f[:,j],1,nxcmp)
         m_rowvec=fct_metric_inp(X,xcmp,pars_metric_inp) #
         inds = m_rowvec .> 0
         m_rowvec = (m_rowvec[inds]).^p
         diffs_f = fct_metric_outp(F,fcmp,pars_metric_outp) - alpha
         diffs_f = diffs_f[inds]
         #diffs_f = abs(sample_f_old-f[i]) - hpa.alpha
         L =max(L,maximum(diffs_f./m_rowvec));
      end
     end
    return L
   end
emp_L2(x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64) = emp_L2(x::Array{Float64,2},f::Array{Float64,2},fct_metric_inp::Function,fct_metric_outp::Function,p=1.::Float64,alpha=0.::Float64,[1.],[1.])
# ===============  Bayesian belief update over L based on Pareto density

density_pareto(x::Float64)= density_pareto(x::Float64,[0.00000000000001,0.])
density_pareto(x::Float64,pars::Vector{Float64}) = density_pareto([x],pars::Vector{Float64})

function density_pareto(x::Array{Float64,1},pars::Vector{Float64})
  #m: min parameter >0
  #nu: shape parameter (nu = 0: uninformative)
  m = pars[1]
  nu=pars[2]
  p=zeros(length(x))
  for i=1:length(x)
    xi = x[i]
    if xi >= m
      p[i]= nu * m.^nu./(xi.^(nu+1))
    end
  end
  return p
end

type HoelParEst_Bayes
  #container for current parameter belief
   pars_density::Vector{Float64}#Parameters of density over best Hoelder constant. first parameter xmin: second: nu
   p::Float64 #hoelder exponent
   D::sample  #Data of the form {(s,f(s),e)}
   fct_metric_inp::Function
   fct_metric_outp::Function
   alpha::Float64
   pars_metric_inp::Vector{Float64}
   pars_metric_outp::Vector{Float64}
end
HoelParEst_Bayes(par) = HoelParEst_Bayes(par,1.,sample(zeros(0,0),zeros(0,0),0.),vecmetric_2norm,vecmetric_2norm,0.)
HoelParEst_Bayes(par,p,D,m1,m2,alpha)=HoelParEst_Bayes(par,p,D,m1,m2,alpha,[1.],[1.])
function append_sample_N_update_L_belief_pareto!(hpa::HoelParEst_Bayes,x::Array{Float64,2},f::Array{Float64,2},e::Float64) # == probably want to use this over append_sample_N_update_L_gen!
   #update L:

   if isempty(hpa.D.inp)
     Lcross = 0
   else
    Lcross=emp_cross_L_between_samples2(hpa.D.inp,hpa.D.outp,x,f,hpa.fct_metric_inp,hpa.fct_metric_outp,hpa.p,hpa.alpha,hpa.pars_metric_inp,hpa.pars_metric_outp)
   end
   l =size(x,2)
   if l <= 1
     Lnew =0
   else
    Lnew =emp_L2(x,f::Matrix{Float64},hpa.fct_metric_inp::Function,hpa.fct_metric_outp::Function,hpa.p::Float64,hpa.alpha,hpa.pars_metric_inp,hpa.pars_metric_outp)
   end
   #update 1st pareto parameter:
   hpa.pars_density[1] =maximum([hpa.pars_density[1];Lcross;Lnew])
   #update 2nd pareto parameter:
   hpa.pars_density[2] += l #add the number of tex to the shape parameter
   #append the new data points:
   append_sample!(hpa.D,x,f,e)
 end



end

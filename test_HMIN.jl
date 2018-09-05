

include("../mymodules/HMIN.jl")

#using HMIN
#grid = collect([ .3 .5] )'
#println(compute_radii_of_sample_grid(grid,0.,1.))

d = 6
a = -1. * ones(d)
b= 2. * ones(d)
L=1.
p=1.
alpha=0.
fct= x-> maximum(abs(x))+1


hopt2 = HMIN.HMIN_multidim(a,b,fct,L)

#@time HMIN.RefineGridNoOfTimes!(hopt2,100)
#m,i=find_min(hopt2)
@time argmin,m,i,counter = HMIN.minimiseUntilErrthresh!(hopt2,.01)

println("The minimum is determined to be: ",m)
println("The minimiser is determined to be: ",argmin)
println("Num. of iterations for minimisation: ",counter)

xgrid = hopt2.D.inp[1,:]'
ygrid = hopt2.D.inp[2,:]'
z = hopt2.D.outp'

using Gadfly
#Gadfly.plot(x=xgrid,y=ygrid)

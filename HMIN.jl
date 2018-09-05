module HMIN
importall Base
export sample,HMIN_multidim,minimiseUntilErrthresh!,RefineGridNoOfTimes!,compute_radii_of_sample_grid,find_min,com_minbnd,minimise_Shubert

type sample
   inp::Array{Float64,2}  #x
   outp::Array{Float64,2} #  \tilde f(x) where tilde f(x) \in [f(x)-e,f(x) +e]
   errbnd::Float64
 end

sample(inp::Vector{Float64},outp::Vector{Float64},errbnd::Float64) = sample(inp'',outp'',errbnd)

function rem_sample_pts!(s::sample,inds::Vector{Int64})
  ss = size(s.inp,2)
  indkeep = setdiff(collect(1:ss),inds)
  s.inp = s.inp[:,indkeep]
  s.outp = s.outp[:,indkeep]
end

type HMIN_multidim #Type for online minimisation of an L-p Hoelder function
    #where the L-P Hoelderness is given relative to the maximum-norm
   a::Vector{Float64}# I = {x : a <= x <= b componentwise}
   b::Vector{Float64}
   fct::Function
   L::Float64#Hoelder constant est
   p::Float64#Hoelder exponent est
   D::sample  #Data of the form {(s,f(s),e)}
 #  fct_metric_inp::Function
 #  fct_metric_outp::Function
   alpha::Float64 #threshold for Lip const est
   gridradii::Matrix{Float64} #radii of the input grid. d x numsamples mat
   indminbnd::Int64 #index of minimum value of sample hyperrect where minbnd occurs
   minbnd::Float64 #min value of floor

 end
HMIN_multidim(a,b,fct,L) = HMIN_multidim(a,b,fct,L,1.)
HMIN_multidim(a,b,fct,L,p) = HMIN_multidim(a,b,fct,L,p,sample(collect((a+b)/2),collect(fct((a+b)/2)),0.),0.)
HMIN_multidim(a,b,fct,L,p,D,alpha) = begin gridradii = compute_radii_of_sample_grid(D.inp,a,b);minbnd,indminbnd =comp_minbnd(L,p,D.outp,gridradii[1]);HMIN_multidim(a,b,fct,L,p,D,alpha,gridradii[1],indminbnd,minbnd-D.errbnd); end


function compute_radii_of_sample_grid(inp::Matrix{Float64},minxvec=[],maxxvec=[])
  #inp[:,j] \leq minxvec componentwise
   #inp[:,j] \leq maxxvec componentwise
    # return ra,rb
    #ra[i,j] = radius in left(neg) direction from sample s_j along dimension i
    #rb[i,j] = radius in right direction from sample s_j along dimension i
  d,m = size(inp)
  if m < 1
    return []
  end
    if isempty(minxvec)
        minxvec = minimum(inp,1)
    end
    if isempty(maxxvec)
        maxxvec= maximum(inp,1)
    end

  ri = zeros(d,m+1) #radii of all samples in ith dimension, ri[i,j] is left radius for jth sample in ith dim
  #similarly, ri[i,j+1] is the right radius of the jth sample in ith dim

  for i=1:d
    v = [minxvec[i] inp[i,:] maxxvec[i]]
    v=sortcols(v)
    v2 = v[2:end]
    ri[i,:] = abs(v2-v[1:end-1])./2.
        ri[i,1] = 2*ri[i,1]
        ri[i,end] = 2*ri[1,end] #radius at fringes of the data extends all the way to the lower and upper bound


  end
  ra = ri[:,1:end-1]
  rb = ri[:,2:end]
    return ra,rb
end

function find_min(self::HMIN_multidim)
    return m,i = findmin(self.D.outp)
end

function comp_minbnd(L::Float64,p::Float64,outp::Matrix{Float64},gridradii::Matrix{Float64})
    rmx = maximum(gridradii,1)
    errbnds = L .* rmx.^(p)
    minbnds = outp - errbnds
    m,i = findmin(minbnds)
    return m,i,errbnds[i]
end

 comp_minbnd(self::HMIN_multidim) = comp_minbnd(self.L,self.p,self.D.outp,self.gridradii)


function comp_minbnd!(self::HMIN_multidim)
    self.minbnd,self.indminbnd,self.errfvalmin = comp_minbnd(self.L,self.p,self.D.outp,self.gridradii)
end

#----------------- grid refinement ---------------------

function SplitHyperrectAlongDim(c,r,m)
        if length(c) >1
       # @bp
            c1 = 1.*c #multiplying with 1 yields assignment by value
            c3 = 1.*c
            rcp=1.*r
            c1[m] = c[m]- (r[m]*2./3.)
            c3[m] = c[m]+ (r[m]*2./3.)
            rcp[m] = r[m] *1./3.
        else
            #c1 = 1.*c #multiplying with 1 yields assignment by value
            #c3 = 1.*c
            ##rcp=1.*r
            c1 = c- (r*2./3.)
            c3 = c+ (r*2./3.)
            rcp = r *1./3.
        end
        return (c1,c,c3),(rcp,rcp,rcp)
      end


     function SelectHyperrect2Split!(self::HMIN_multidim,method="minbnd")


        if method =="minbnd"
        rmx = maximum(self.gridradii,1)
        errbnds = self.L .* rmx.^(self.p)
        minbnds = self.D.outp - errbnds
        self.minbnd,self.indminbnd = findmin(minbnds)
          return self.indminbnd

        elseif method =="rndhyperrect"
             return rand(1:size(self.gridradii,2))
         else
             return []
         end
     end

function RefineGrid!(self::HMIN_multidim,fct=[],method="minbnd")
        #fct: function to be estimated/integrated/sampled

        #l = len(grid.r) # last element at l-1
         if isempty(fct)
            fct = self.fct
         end
         ind = SelectHyperrect2Split!(self,method)
         c = self.D.inp[:,ind]
         r = self.gridradii[:,ind]
         mx,m = findmax(r)  # dim with maximum radius

         c_list,r_list = SplitHyperrectAlongDim(c,r,m)
         #@bp
         #get the new function values:
         f_list = [fct(c_list[1]) fct(c_list[3])]
         #next, we update the grid...
         self.D.outp = [self.D.outp f_list]
         #import pdb; pdb.set_trace()
        self.gridradii[:,ind] = r_list[2]#update the old point by shrunken radii

         #grid.r =np.append(grid.r,[r_list[0],r_list[2]])
         self.gridradii = [self.gridradii r_list[1] r_list[3]]
         #grid.x = np.append(grid.x,[c_list[0],c_list[2]])
         self.D.inp=[self.D.inp c_list[1] c_list[3]]

       end



# # ===================================================================================

# ==== Shubert's minimisation for 1d-Lipschitz functions

function minimise_Shubert(fct,I = Vector{Float64},L=1.,errthresh =0.0001,maxevals=10000)
  #Lipschhitz optimisation a la Shubert
  #fct: function handle
  #I: domain interval
  #L: Lipschitz constant
  grid_t = [I[1];I[2]]

  grid_fctvals = [ fct(grid_t[1]);fct(grid_t[2])]

  n = 2
    argmins_floor = zeros(n-1);
    minvals_floor = zeros(n-1);
    for i =1:n-1
                argmins_floor[i] = (grid_t[i] +grid_t[i+1])/2 - (grid_fctvals[i+1]-grid_fctvals[i])/(2*L);
                minvals_floor[i] = (grid_fctvals[i+1]+grid_fctvals[i])/2 - L * (grid_t[i+1] -grid_t[i]);
    end
    curr_fmin = minimum(grid_fctvals)
    m,ind = findmin(minvals_floor)

  xi = argmins_floor[ind]
    fxi = fct(xi)
    err = abs(curr_fmin - fxi)
  while n < maxevals && err > errthresh

    #refine grid:
    ind_crit = findfirst(grid_t .> xi) #xi should never be smaller than the smallest nor larger than the largest element in grid_t
   #splice!(grid_t,ind_crit,xi)
   splice!(grid_t,ind_crit,[xi;grid_t[ind_crit]])
    splice!(grid_fctvals,ind_crit,[fxi;grid_fctvals[ind_crit]])
     #(rem: the new value in the grid now is at pos crit_ind in the grid)

    #refine floor:
     i = ind_crit-1
     newxileft = (grid_t[ind_crit-1] +grid_t[ind_crit])/2 - (grid_fctvals[ind_crit]-grid_fctvals[ind_crit-1])/(2*L);
     newxiright = (grid_t[ind_crit] +grid_t[ind_crit+1])/2 - (grid_fctvals[ind_crit+1]-grid_fctvals[ind_crit])/(2*L);
     splice!(argmins_floor,i,[newxileft;newxiright])

     newfloorxileft = (grid_fctvals[ind_crit]+grid_fctvals[ind_crit-1])/2 - L * (grid_t[ind_crit] -grid_t[ind_crit-1]);
     newfloorxiright = (grid_fctvals[ind_crit+1]+grid_fctvals[ind_crit])/2 - L * (grid_t[ind_crit+1] -grid_t[ind_crit]);
     splice!(minvals_floor,i, [newfloorxileft;newfloorxiright])
     n+=1
     curr_fmin = minimum(grid_fctvals)
     m,ind = findmin(minvals_floor)
     xi = argmins_floor[ind]
     fxi = fct(xi)
     err = abs(curr_fmin - fxi)

  end
  curr_fmin,i = findmin(grid_fctvals)
  curr_argmin = grid_t[i]
  return curr_argmin,curr_fmin,n
end

# ==== Shubert's minimisation for 1d-Lipschitz functions  -- ende

function RefineGridNoOfTimes!(self::HMIN_multidim,nooftimes::Int64,fct=[],method="minbnd")
         for i=1:nooftimes
             RefineGrid!(self,fct,method)
         end
       end



function minimiseUntilErrthresh!(self::HMIN_multidim,errthresh::Float64,maxiter =1000000,fct=[],method="minbnd")
    m,i=find_min(self)
    counter = 0
   # @bp
    while (abs(m-self.minbnd) >=errthresh) & (counter <= maxiter)
             RefineGrid!(self,fct,method)
             counter = counter+1
        m,i=find_min(self)
    end
    argmin = self.D.inp[:,i]
    return argmin,m,i,counter
 end


end

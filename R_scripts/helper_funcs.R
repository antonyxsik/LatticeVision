library(LatticeKrig)
library(spam64)
library(tictoc)
library(rhdf5)
library(maps)
library(cmocean)
library(here)
rm(list=ls())


###########################################
######     Helper Functions          #####
###########################################

# Simulation function for generating realizations of the SAR
LKrig.sim <- function(x1, LKinfo, M = 1, just.coefficients = FALSE) {
  Q <- LKrig.precision(LKinfo)
  Qc <- chol(Q, memory = LKinfo$choleskyMemory)
  m <- LKinfo$latticeInfo$m
  E <- matrix(rnorm(M * m), nrow = m, ncol = M)
  randomC <- backsolve(Qc, E)
  if (just.coefficients) {
    return(randomC)
  } 
  else {
    PHI1 <- LKrig.basis(x1, LKinfo)
    return(PHI1 %*% randomC)
  }
}	


# LKrig function that creates the precision matrix
# which requires the creation of the SAR matrix 
LKrig.precision <- function(LKinfo, return.B = FALSE,
                            verbose=FALSE) { 
  L <- LKinfo$nlevel
  offset <- LKinfo$latticeInfo$offset
  # some checks on arguments
  LKinfoCheck(LKinfo)
  # ind holds non-zero indices and ra holds the values
  ind <- NULL
  ra <- NULL
  da <- rep(0, 2)
  # loop over levels
  for (j in 1:L) {
    # evaluate the SAR matrix at level j.
    tempB<- LKrigSAR( LKinfo, Level=j)
    if( verbose){
      cat("dim indices in spind of B:",dim( tempB$ind) , fill=TRUE)            	
    }
    # accumulate the new block
    # for the indices that are not zero
    ra <- c(ra,tempB$ra )
    ind <- rbind(ind, tempB$ind + offset[j])
    # increment the dimensions
    da[1] <- da[1] + tempB$da[1]
    da[2] <- da[2] + tempB$da[2]
  }
  # dimensions of the full matrix
  # should be da after loop
  # check this against indices in LKinfo
  #
  if ((da[1] != offset[L + 1]) | (da[2] != offset[L + 
                                                  1])) {
    stop("Mismatch of dimension with size in LKinfo")
  }
  # convert to spind format:
  # tempBtest <- list(ind = ind, ra = ra, da = da) 
  # tempB<- spind2spam( tempBtest)
  if( verbose){
    cat("dim of ind (fullB):", dim( ind), fill=TRUE)
  }
  tempB <- spam( list( ind=ind, ra), nrow=da[1], ncol=da[2])
  if( verbose){
    cat("dim after spind to spam in precision:", dim( tempB), fill=TRUE)
  }
  if (return.B) {
    return(tempB)
  }
  else {
    # find precision matrix Q = t(B)%*%B and return
    return(t(tempB) %*% (tempB))
  }
}


# returns sparse SAR matrix which is used in the 
# precision function, which is then called in the 
# simulation function. here we ensure the B matrix
# has periodic properties which allows us to 
# respect the Mercator projection

LKrigSAR <- function( object, Level, ...){ 
  
  mx1<-              object$latticeInfo$mx[Level,1]
  mx2<-              object$latticeInfo$mx[Level,2]
  m<- mx1*mx2
  #
  a.wght<- (object$a.wght)[[Level]]
  
  stationary <-     (attr( object$a.wght, "stationary"))[Level]
  first.order<-     attr( object$a.wght, "first.order")[Level]
  isotropic  <-     attr(object$a.wght, "isotropic")[Level]
  distance.type <-  object$distance.type
  if( all(stationary & isotropic) ) {
    if( any(unlist(a.wght) < 4) ){
      stop("a.wght less than 4")
    }
  }
  
  #  either  a.wght is a  matrix (rows index lattice locations)
  #  or fill out matrix of this size with stationary values
  dim.a.wght <- dim(a.wght)
  
  # figure out if just a single a.wght or matrix is passed
  # OLD see above first.order <-  (( length(a.wght) == 1)|( length(dim.a.wght) == 2)) 
  
  # order of neighbors and center
  index <- c(5, 4, 6, 2, 8, 3, 9, 1, 7)
  # dimensions of precision matrix
  da <- as.integer(c(m, m))
  # contents of sparse matrix organized as a 2-dimensional array
  # with the second dimension indexing the weights for center and four nearest neighbors.
  if (first.order) {
    ra <- array(NA, c(mx1*mx2, 5))
    ra[,  1] <- a.wght
    ra[,  2:5] <- -1
  }
  else {
    ra <- array(NA, c(mx1 * mx2, 9))
    for (kk in 1:9) {
      # Note that correct filling happens both as a scalar or as an mx1 X mx2 matrix
      if (stationary) {
        ra[ , kk] <- a.wght[index[kk]]
      }
      else {
        ra[ ,  kk] <- a.wght[ , index[kk]]
      }
    }
  }
  #
  #  Order for 5 nonzero indices is: center, top, bottom, left right
  #  a superset of indices is used to make the arrays regular.
  #  and NAs are inserted for positions beyond lattice. e.g. top neighbor
  #  for a lattice point on the top edge. The NA pattern is also
  #  consistent with how the weight matrix is filled.
  #
  Bi <- rep(1:m, 5)
  i.c <- matrix(1:m, nrow = mx1, ncol = mx2)
  # indices for center, top, bottom, left, right or ... N, S, E, W
  # NOTE that these are just shifts of the original matrix
  Bj <- c(i.c,
          LKrig.shift.matrix(i.c, 0, -1), # top 
          LKrig.shift.matrix(i.c, 0,  1), #bottom
          LKrig.shift.matrix(i.c, 1,  0, periodic = TRUE), #left (periodic here)
          LKrig.shift.matrix(i.c, -1, 0, periodic = TRUE)  #right (periodic here)
  )
  # indices for NW, SW, SE, SW
  if (!first.order) {
    Bi <- c(Bi, rep(1:m, 4))
    Bj <- c(Bj,
            LKrig.shift.matrix(i.c,  1,  1, periodic = c(TRUE,TRUE)), # bottom left (periodic here)
            LKrig.shift.matrix(i.c, -1,  1, periodic = c(TRUE,TRUE)), # bottom right (periodic here)
            LKrig.shift.matrix(i.c,  1, -1, periodic = c(TRUE,TRUE)), # top left (periodic here)
            LKrig.shift.matrix(i.c, -1, -1, periodic = c(TRUE,TRUE)) # top right (periodic here)
            # LKrig.shift.matrix(i.c,  1,  1), # bottom left 
            # LKrig.shift.matrix(i.c, -1,  1), # bottom right 
            # LKrig.shift.matrix(i.c,  1, -1), # top left 
            # LKrig.shift.matrix(i.c, -1, -1) # top right 
    )
  }
  
  # find all cases that are actually in lattice
  good <- !is.na(Bj)
  # remove cases that are beyond the lattice and coerce to integer
  # also reshape ra as a vector stacking the 9 columns
  #
  Bi <- as.integer(Bi[good])
  Bj <- as.integer(Bj[good])
  ra <- c(ra)[good]
  # return spind format because this is easier to accumulate
  # matrices at different multiresolution levels
  # see calling function LKrig.precision
  #
  # below is an add on hook to normalize the values to sum to 4 at boundaries
  if(!is.null(object$setupArgs$BCHook)){
    M<- da[1]
    for( i in 1: M){
      rowI<- which(Bi== i)
      rowNN<- rowI[-1]
      ra[rowNN]<-  4*ra[rowNN] / length(rowNN )
    }
    
  }
  return(list(ind = cbind(Bi, Bj), ra = ra, da = da))
}



# for making the SAR quickly 
LKrigSARGaussian<- function(LKinfo,M,
                            asList=FALSE ){
  coefSAR<- NULL
  nLevel<- LKinfo$nlevel
  for( K in 1:nLevel){
    B <- LKrigSAR(LKinfo,Level=K)
    B<- spind2spam(B) # convert to spam format 
    mLevel<- LKinfo$latticeInfo$mLevel[K]
    # here using Gaussian but can change to other distribution 
    U<- runif(mLevel*M )
    E<-  qnorm(U)
    E<- matrix(E, mLevel, M)
    #
    y<- as.matrix(solve(B, E)) # uses sparse methods
    if(!asList){
      coefSAR<- rbind(coefSAR, y)
    }
    else{ coefSAR<- c( coefSAR, list(y))}
  }
  return( coefSAR)
}


# function for defining fun coastline-like surfaces
coastline <- function(x, coast_coef, coast_bump_scale, coast_freq) {
  return(coast_coef * x + coast_bump_scale * sin(2 * pi * coast_freq * x) 
         + 0.01 * rnorm(length(x)))
}

# read current dims of h5 data
get_dims <- function() {
  info  <- h5ls(file_name, all=TRUE)
  row   <- subset(info, name==dataset_name & otype=="H5I_DATASET")
  dims  <- as.integer(strsplit(row$dim, " x ")[[1]])
  names(dims) <- c("nx","ny","nz")
  dims
}

# predict constant values
predict.constantValue<- function(object,x){
  n<- length(object$values)
  m<- nrow( x)
  return( matrix( object$values, nrow=m, ncol=n, byrow=TRUE ) )
}

# predict functions for param grids 
predict.surfaceGrid<- function(object,x){
  interp.surface( object, x)
}

# predict function for non-stationary, anisotropic awght
predict.multivariateSurfaceGrid<- function(object,x){
  dimZ<- dim( object$z)
  L<- dimZ[3]
  out<- matrix( NA, nrow= nrow(x), ncol=L)
  for (  l in 1:L){
    out[,l]<- interp.surface( 
      list( x=object$x,y=object$y, z=object$z[,,l]) , x)
  }
  return( out)
}


# param field generator for pattern figs
generate_param_field <- function(
    paramtype, 
    config, 
    sGrid, 
    rows, 
    columns,
    n){
  
  if (paramtype == "awght"){
    # sampling from predetermined awghts 
    # (this can be changed, just a design choice)
    lower_bound <- min(awghts)
    upper_bound <- max(awghts)
    midpoint <- (lower_bound + upper_bound)/2 # 5
    
    param_constant <- sample(awghts, 1)
    param_low <- sample(low_awghts, 1)
    param_high <- sample(high_awghts, 1)
    
    # these variables are dependent on the domain of the param (awght)
    gauss_amps <- runif(2, 0.1, 0.5) * ifelse(param_constant <= midpoint, 1, -1)
    mult_factor <- runif(1, 0.001, 0.1998)
    
  } else if (paramtype == "rho"){
    # rho is constructed simply based on bounds
    lower_bound <- 1
    upper_bound <- 7
    midpoint <- (lower_bound + upper_bound) / 2  # 4
    
    param_constant <- runif(1, lower_bound, upper_bound)
    param_low <- runif(1, lower_bound, param_constant)
    param_high <- runif(1, param_constant, upper_bound)
    
    # dependent on rho domain
    gauss_amps <- runif(2, 0.1, 1.5) * ifelse(param_constant <= midpoint, 1, -1) 
    mult_factor <- runif(1, 0.001, 0.7498)  
    
  } else if (paramtype == "theta"){
    # making sure to obtain all possible ellipses
    lower_bound <- 0
    upper_bound <- 3
    midpoint <- (lower_bound + upper_bound) / 2  # 1.5 (roughly pi/2)
    
    param_constant <- runif(1, lower_bound, upper_bound)
    param_low <- runif(1, lower_bound, param_constant)
    param_high <- runif(1, param_constant, upper_bound)
    
    # dependent on theta domain
    gauss_amps <- runif(2, 0.1, pi/4) * ifelse(param_constant < midpoint, 1, -1) 
    mult_factor <- runif(1, 0.001, 0.9998) 
    
  } else {
    stop("Unknown paramtype. Please use either awght, rho, or theta.")
  }
  
  # these quantities dont depend on which param field we are making
  taper_sd <- runif(1, 0.05, 1)
  
  gauss_slopes <- runif(2, 0.2, 0.5)
  # gauss locations need to be within [-1,1],[-1,1] domain
  gauss_locs <- runif(4, sGrid[1], tail(sGrid,1)[1]) 
  
  coast_sharpnesses <- runif(2, 3, 50)
  coast_bump_scales <- runif(2, 0.1, 0.5)
  coast_freqs <- runif(2, 0.4, 3)
  coast_coefs <- runif(2, -2 , 2)
  # make sure that adding coastlines doesnt go out of bounds
  coast_amp1 <- runif(1,0.1 , 0.9)
  coast_amp2 <- runif(1, 0.1, 1-coast_amp1)
  
  # make sure sin amplitude doesnt go out of bounds
  if (param_constant > midpoint){
    sin_amp <- runif(1, 0, upper_bound - param_constant)
  }
  else{
    sin_amp <- runif(1, 0, param_constant - lower_bound)
  }
  sin_freq <- runif(1, 1.5, 5)
  sin_orientation <- sample(c("horiz","vert"),1)
  
  #number of basis for generating a gp for the param field. 
  # a gp will then be generated from this gp. 
  #Just like inception...
  num_basis_param <- sample(c(6:32), 1) 
  
  if (config == "constant"){
    param_func <- rep(param_constant, n)
  }
  
  else if (config == "taper"){
    taper<- pnorm( sGrid[,1] + sGrid[,1],
                   mean = 0, sd = taper_sd)
    param_func<- param_low*taper +  param_high*(1-taper)
  }
  
  else if (config == "Gaussian"){
    param_func <- param_constant + 
      gauss_amps[1] * exp(-((sGrid[,1]^2 + sGrid[,2]^2) / gauss_slopes[1])) 
  }
  
  else if (config == "coastline"){
    param_func <- param_low + (param_high - param_low) / 
      (1 + exp(-coast_sharpnesses[1] * (sGrid[,2] - coastline(sGrid[,1], 
                                                              coast_coefs[1],
                                                              coast_bump_scales[1],
                                                              coast_freqs[1]))))
  }
  
  else if (config == "sinwave"){
    if (sin_orientation == "vert"){
      param_func <- param_constant + sin_amp * sin( pi * sGrid[,1] * sin_freq)
    }
    else{
      param_func <- param_constant + sin_amp * cos( pi * sGrid[,2] * sin_freq)
    }
  }
  
  else if (config == "double_Gaussian"){
    
    peak1 <- gauss_amps[1] * exp(-((sGrid[,1] - gauss_locs[1])^2 + 
                                     (sGrid[,2] - gauss_locs[2])^2) / gauss_slopes[1])
    peak2 <- gauss_amps[2] * exp(-((sGrid[,1] + gauss_locs[3])^2 + 
                                     (sGrid[,2] + gauss_locs[4])^2) / gauss_slopes[2])
    param_func <- param_constant + peak1 + peak2
  }
  
  else if (config == "double_coastline"){
    coastline1 <- (param_high - param_low)/
      (1 + exp(-coast_sharpnesses[1] * (sGrid[,2] - 
                                          coastline(sGrid[,1], 
                                                    coast_coefs[1], 
                                                    coast_bump_scales[1],
                                                    coast_freqs[1]))))
    coastline2 <- (param_high - param_low)/
      (1 + exp(-coast_sharpnesses[2] * (sGrid[,2] - 
                                          coastline(sGrid[,1], 
                                                    coast_coefs[2], 
                                                    coast_bump_scales[2],
                                                    coast_freqs[2]))))
    param_func <- param_low + (coast_amp1 * coastline1) + (coast_amp2 * coastline2)
  }
  
  else if (config == "GP_gp") {
    # pick scaling strategy half the time
    scale_choice <- sample(c("minmax","perturb"), size = 1, prob = c(0.5,0.5))
    
    # simulate the little GP
    LKinfo_param <- LKrigSetup(
      sGrid,
      NC        = num_basis_param,
      nlevel    = 1,
      a.wght    = sample(awghts, 1),
      nu        = 1,
      normalize = FALSE
    )
    f_param <- LKrig.basis(sGrid, LKinfo_param) %*% 
      LKrigSARGaussian(LKinfo_param, M = 1)
    
    if (scale_choice == "minmax") {
      # full min/max rescale to [lower_bound, upper_bound]
      f_min    <- min(f_param)
      f_max    <- max(f_param)
      f_norm   <- (f_param - f_min) / (f_max - f_min)
      param_func <- f_norm * (upper_bound - lower_bound) + lower_bound
      
    } else {
      # old small perturbation around param_constant technique 
      f_param <- (f_param - min(f_param)) / (max(f_param) - min(f_param))
      if (param_constant < midpoint) {
        param_func <- ((f_param * mult_factor) + 1) * param_constant
      } else {
        param_func <- (1 - (f_param * mult_factor)) * param_constant
      }
    }
  }
  
  # finally, reshape back to a matrix and return
  param_field <- matrix(param_func, nrow = rows, ncol = columns)
  return(param_field)
}


# function to plot model estimated params
plot_params <- function(
    kappa2,
    theta, 
    rho, 
    border_lwd = 1
){
  awght <- kappa2 + 4
  # sanity plotting for input and output params
  par(mfrow = c(2,2), mar =   c(3.1, 4.1, 2.1, 2.1))
  imagePlot(x = lon, y = lat, 
            first_field_norm, main = "First Clim Field", col = turbo(256)) #, 
  #horizontal = TRUE)
  map("world2", add = TRUE, 
      col = "grey0", lwd = border_lwd)
  
  imagePlot(x = lon, y = lat,
            awght, main = "awght", col = viridis(256))
  map("world2", add = TRUE, 
      col = "grey90", lwd = border_lwd)
  
  imagePlot(x = lon, y = lat,
            theta, main = "Theta", col = viridis(256))
  map("world2", add = TRUE, 
      col = "grey90", lwd = border_lwd)
  
  imagePlot(x = lon, y = lat,
            rho, main = "Rho", col = viridis(256))
  map("world2", add = TRUE, 
      col = "grey90", lwd = border_lwd)
  par(mfrow = c(1,1))
}

# function for plotting covariance surface with contours
plot_cov_surface <- function(
    LKinfo, 
    real_field,
    loc_x, 
    loc_y, 
    plotting = TRUE
){
  cov_surface <- LKrig.cov(sGrid, rbind(c(loc_x,loc_y)), LKinfo)
  cov_surface <- matrix(cov_surface, nrow = rows, ncol = columns)
  
  if (plotting == TRUE){
    par(mfrow = c(1,2), mar=  c(5.1, 4.1, 4.1, 2.1))
    image.plot(as.surface(sGrid, cov_surface), col = viridis(256), 
               main = "Correlation")
    contour( as.surface( sGrid, cov_surface), col = "white", add = TRUE)
    
    imagePlot(as.surface(sGrid, real_field), 
              main = "First Clim Field", col = turbo(256))
    contour( as.surface( sGrid, cov_surface), col = "grey30", add = TRUE)
    par(mfrow = c(1,1))
  }
  
  return (cov_surface)
}


# plots the real and simulated fields side by side 
plot_real_sim <- function(
    real_field, 
    simulated_field,
    border_lwd = 1
){
  # plot the real field and simulated field
  par(mfrow = c(1,2), mar=  c(5.1, 4.1, 4.1, 2.1))
  imagePlot(x = lon, y = lat,
            real_field, main = "", col = turbo(256)) #Clim Field
  map("world2", add = TRUE, 
      col = "grey30", lwd = border_lwd)
  
  image.plot(x = lon, y = lat,
             simulated_field, col = turbo(256), main = "") #Simulated field
  map("world2", add = TRUE, 
      col = "grey30", lwd = border_lwd)
  par(mfrow = c(1,1))
}

# function for centering plots in the pacific
pacific_centering <- function(fields){
  if (length(dim(fields)) == 3){
    fields <- fields[c((rows/2+1):rows,1:(rows/2)),,]
  }
  else if (length(dim(fields)) == 2){
    fields <- fields[c((rows/2+1):rows,1:(rows/2)),]
  }
  else {
    stop("Input must be a 2D or 3D array")
  }
}

# MAIN GENERATION FUNCTION
# takes in the estimated parameters and creates synthetic fields
generate_synthetic_reps <- function(
    kappa2, 
    theta, 
    rho,
    rhox,
    rhoy,
    n_replicates = 30,
    random_seed = 777,
    smooth_choice = FALSE,
    normalize = TRUE
){
  
  # create H tensor out of params
  H11 <- ( rhox^2 * (cos(theta))^2) + ( rhoy^2 * (sin(theta))^2 ) 
  H12 <- (rhoy^2 - rhox^2)*(sin(theta)*cos(theta))
  H21 <- H12 
  H22 <- (rhox^2 * (sin(theta))^2) + (rhoy^2 * (cos(theta))^2)
  
  # fill the high dimensional stencil (9 fields)
  stencil_tensor <- array( NA, c( rows,columns,9))
  stencil_tensor[,,1] <- 0.5 * H12
  stencil_tensor[,,2] <- -H22
  stencil_tensor[,,3] <- -0.5 * H12
  stencil_tensor[,,4] <- -H11
  stencil_tensor[,,5] <- kappa2 + 2 * H11 + 2 * H22
  stencil_tensor[,,6] <- -H11
  stencil_tensor[,,7] <- -0.5 * H12
  stencil_tensor[,,8] <- -H22
  stencil_tensor[,,9] <- 0.5 * H12
  
  # put everything into awght obj of a particular class
  awght_obj <- list( x= gridList$x,  y= gridList$y, z=stencil_tensor )
  class( awght_obj)<- "multivariateSurfaceGrid"
  
  #setup an LKinfo object for generating fields 
  set.seed(random_seed)
  LKinfo <- LKrigSetup(sGrid, NC =rows,
                       nlevel = 1, 
                       a.wghtObject =  awght_obj, 
                       normalize=FALSE, 
                       NC.buffer = 0, overlap = 2.5, nu = 1) 
  
  
  # generate fields
  gen_time <- system.time(
    f <- LKrig.sim( sGrid, LKinfo, M = n_replicates, just.coefficients = smooth_choice)
  )
  
  # normalize the fields 
  if (normalize == TRUE){
    mu <- rowMeans(f)
    sd <- apply(f, 1, sd) 
    f <- (f - mu)/sd
  }
  
  # SAVE 
  object <- list()
  object$gen_time <- gen_time
  object$LKinfo <- LKinfo
  object$f <- f
  object$mu <- mu
  object$sd <- sd
  
  return (object)
}

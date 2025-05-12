library(LatticeKrig)
library(spam64)
library(tictoc)
library(rhdf5)
library(maps)
library(moments)
library(nortest)
library(ggplot2)
library(ggpubr)
library(dplyr)
library(scico)
library(cmocean)
library(RColorBrewer)
theme_set(theme_pubr())


rm(list=ls())
setwd("C:/Users/anton/Desktop/Research/Paper_3/LatticeVision")

# Load in modified LK files 
source(
  "C:/Users/anton/Desktop/Research/Paper_3/LatticeVision/R_data_generation/LKrig.sim.mercator.R"
)

###########################################
######     Helper Functions          #####
###########################################
# many of these functions are simply written
# for plotting convenience
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
    # changed quantities: (0.5, 0.1998)
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
    
    # dependent on rho domain. changed quantities:  (1.5, 0.7498)
    gauss_amps <- runif(2, 0.1, 1.5) * ifelse(param_constant <= midpoint, 1, -1) 
    mult_factor <- runif(1, 0.001, 0.7498) 
    
  } else if (paramtype == "theta"){
    # full range of correlation ellipses
    lower_bound <- 0
    upper_bound <- 3
    midpoint <- (lower_bound + upper_bound) / 2  # 1.5 (roughly pi/2)
    
    param_constant <- runif(1, lower_bound, upper_bound)
    param_low <- runif(1, lower_bound, param_constant)
    param_high <- runif(1, param_constant, upper_bound)
    
    # dependent on theta domain. changed quantities: (pi/8, 0.4998)
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
  sin_orientation <- "horiz" #sample(c("horiz","vert"),1)
  
  #number of basis for generating a gp for the param field. 
  # a gp will then be generated from this gp. 
  # inception vibes
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
  
  else if (config == "GP_gp"){
    # making param fields from GPs themselves!!
    LKinfo_param <- LKrigSetup(sGrid, 
                               NC = num_basis_param, 
                               nlevel = 1, 
                               a.wght = sample(awghts, 1),
                               nu = 1, 
                               normalize = FALSE)
    
    # faster way to simulate the field 
    f_param <- LKrig.basis(sGrid, LKinfo_param) %*% LKrigSARGaussian(LKinfo_param, M = 1)
    
    
    # minmax 0-1 scale the field
    f_param <- (f_param - min(f_param))/(max(f_param) - min(f_param))
    
    #multiplying factor based on param domain bounds
    if (param_constant < midpoint){
      param_func <- ((f_param * mult_factor) + 1) * param_constant
    }else{
      param_func <- (1 - (f_param * mult_factor)) * param_constant
    }
  }
  
  # param_field <- as.surface( sGrid, param_func)$z
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

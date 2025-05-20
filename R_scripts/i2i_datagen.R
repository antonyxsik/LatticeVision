##########             IMPORTANT              ###########
#-------------------------------------------------------#
# Navigate to "Session" in the top left of RStudio and
# click on "Set Working Directory" and then 
# choose "To Source File Location". 
# Now you can run the script. 
#-------------------------------------------------------#

library(LatticeKrig)
library(spam64)
library(tictoc)
library(rhdf5)
library(here)
# helper functions
source(here("R_scripts", "helper_funcs.R"))


# Some choices to make: 
# total number of simulated fields 
total_sims <- 8000
# chunks the data will be created in (RAM constraints), 1000 is about 16 GB
chunk_size <- 1000
# Other options can be tweaked in the
# "Experimental Setup and Hyperparameters" section below. 


###########################################
######     Param Field Generator     ######
###########################################

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

###########################################
######      Full Data Generator      ######
###########################################

generate_i2i_data <- function(
    N_SIMS, 
    n_replicates = 30, 
    n_buffer = 10,
    random_seed = 777, 
    configs,
    awghts, 
    low_awghts, 
    high_awghts,
    rows = 288,
    columns = 192,
    verbose = FALSE, 
    sanity_plotting = FALSE){
  
  dataset <- array(data = NA, dim = c( ( rows * columns), n_replicates + 3, N_SIMS))
  print("Dataset dimensions:") 
  print(dim(dataset))
  
  rows <- rows + (2*n_buffer)
  columns <- columns + (2*n_buffer)
  n <- rows * columns
  
  #grid for data 
  gridList<- list( x= seq( 1,rows,length.out= rows),
                   y= seq( 1,columns,length.out= columns) )
  sGrid<- make.surface.grid(gridList)
  
  #grid for param fields 
  gridList_param<- list( x= seq( -1,1,length.out= rows),
                         y= seq( -1,1,length.out= columns) )
  sGrid_param<- make.surface.grid(gridList_param)
  
  script_time <- system.time(
    for (sim in 1:N_SIMS){
      # set.seed(random_seed + sim)
      
      # choose configs for params
      config_k <- sample(configs, 1)
      config_th <- sample(configs, 1)
      config_r <- sample(configs, 1)
      
      # make kappa field (with occasional stacking)
      stack_choice_k <- sample(c("yes", "no"), 1)
      if (stack_choice_k == "no") {
        kappa2 <- generate_param_field(
          paramtype = "awght",
          config    = config_k,
          sGrid     = sGrid_param,
          rows      = rows,
          columns   = columns,
          n         = n
        ) - 4
        config_k_2 <- NA
      } else {
        config_k_2   <- sample(configs, 1)
        kappa2_pt1   <- generate_param_field("awght", config_k,   sGrid_param, rows, columns, n) - 4
        kappa2_pt2   <- generate_param_field("awght", config_k_2, sGrid_param, rows, columns, n) - 4
        wght_k       <- runif(1, 0.1, 1)
        kappa2       <- (wght_k * kappa2_pt1) + ((1 - wght_k) * kappa2_pt2)
        rm(kappa2_pt1, kappa2_pt2)
      }
      
      # theta field (with stacking option)
      stack_choice_th <- sample(c("yes", "no"), 1)
      if (stack_choice_th == "no") {
        theta      <- generate_param_field(
          paramtype = "theta",
          config    = config_th,
          sGrid     = sGrid_param,
          rows      = rows,
          columns   = columns,
          n         = n
        )
        config_th_2 <- NA
      } else {
        config_th_2 <- sample(configs, 1)
        theta_pt1   <- generate_param_field("theta", config_th,   sGrid_param, rows, columns, n)
        theta_pt2   <- generate_param_field("theta", config_th_2, sGrid_param, rows, columns, n)
        wght_th     <- runif(1, 0.1, 1)
        theta       <- (wght_th * theta_pt1) + ((1 - wght_th) * theta_pt2)
        rm(theta_pt1, theta_pt2)
      }
      
      # rho field (with stacking option)
      stack_choice_r <- sample(c("yes", "no"), 1)
      if (stack_choice_r == "no") {
        rho        <- generate_param_field(
          paramtype = "rho",
          config    = config_r,
          sGrid     = sGrid_param,
          rows      = rows,
          columns   = columns,
          n         = n
        )
        config_r_2  <- NA
      } else {
        config_r_2 <- sample(configs, 1)
        rho_pt1    <- generate_param_field("rho", config_r,   sGrid_param, rows, columns, n)
        rho_pt2    <- generate_param_field("rho", config_r_2, sGrid_param, rows, columns, n)
        wght_r     <- runif(1, 0.1, 1)
        rho        <- (wght_r * rho_pt1) + ((1 - wght_r) * rho_pt2)
        rm(rho_pt1, rho_pt2)
      }
      
      # derive rhox and rhoy 
      rhox <- sqrt(rho)
      rhoy <- 1 / rhox
      
      if (sanity_plotting == TRUE){
        par(mfrow = c(2,2))
        image.plot(kappa2, col = viridis(256), main = "Kappa2 field")
        image.plot(theta, col = viridis(256), main = "Theta field")
        image.plot(rhox, col = viridis(256), main = "Rhox field")
        image.plot(rhoy, col = viridis(256), main = "Rhoy field")
        par(mfrow = c(1,1))
      }
      
      # populate the H tensor (4 fields)
      # in the case of a constant stencil, this acts as a 2x2 dispersion matrix
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
      
      # remove from memory for space
      rm(H11, H12, H21, H22, stencil_tensor)
      
      if (sanity_plotting == TRUE){
        par(mfrow = c(3,3))
        labels = c("top left corner", "top middle", "top right corner",
                   "middle left", "middle (awght)", "middle right",
                   "bottom left corner", "bottom middle", "bottom right corner")
        for (i in 1:9){
          image.plot(awght_obj$z[,,i], main = labels[i], col = viridis(256))
        }
        par(mfrow = c(1,1))
      }
      
      # setup LKrig object 
      # we make our own buffer here so the package buffer is set to 0
      LKinfo <- LKrigSetup(sGrid, NC =rows,
                           nlevel = 1, 
                           a.wghtObject =  awght_obj, 
                           normalize=FALSE, 
                           NC.buffer = 0, overlap = 2.5) 
      #NOTE: 
      # overlap can be set to anything between 0.001 and 1, it will be same (SAR)
      # once u get above 1 (like 1.001) you start to get basis function smoothing 
      # Although this does not matter for us here, as we will not use basis functions
      # we choose to operate with only the coefficients, which is why we select the 
      # just.coefficients = TRUE option below. This is equivalent to just a SAR!
      
      if (verbose) {
        cat(paste0(
          "Loop #", sim, "\n",
          "  kappa2 stack?     ", stack_choice_k,       "\n",
          "    config1:        ", config_k,             "\n",
          "    config2:        ", config_k_2,           "\n",
          "  theta stack?      ", stack_choice_th,      "\n",
          "    config1:        ", config_th,            "\n",
          "    config2:        ", config_th_2,          "\n",
          "  rho   stack?      ", stack_choice_r,       "\n",
          "    config1:        ", config_r,             "\n",
          "    config2:        ", config_r_2,           "\n\n"
        ))
      }
      
      print(paste0("Loop #", sim))
      
      # simulate the fields
      f <- LKrig.sim( sGrid, LKinfo, M = n_replicates, just.coefficients = TRUE)
      mu <- rowMeans(f)
      sd <- apply(f, 1, sd) 
      # normalize all of the fields
      f <- (f - mu)/sd
      
      if (sanity_plotting == TRUE){
        par(mfrow = c(1,2))
        image.plot(as.surface(sGrid, f[,1]), col = turbo(256), main = "First field")
        
        cov_surface <- LKrig.cov(sGrid, rbind(c(rows/2,columns/2)), LKinfo)
        image.plot( as.surface( sGrid, cov_surface), col = magma(256), 
                    main = paste("Covariance at", rows/2, ",", columns/2))
        contour( as.surface( sGrid, cov_surface), col = "white", add = TRUE)
        
        par(mfrow = c(1,1))
      }
      
      # do some reshaping to trim off the buffer (crust of thickness n_buffer)
      f <- array(f, dim = c(rows, columns, n_replicates))
      f <- f[(n_buffer + 1):(rows - n_buffer), 
             (n_buffer + 1):(columns - n_buffer), ]
      f <- array(f, dim = c( (rows - (2*n_buffer))*(columns - (2*n_buffer)), n_replicates) )
      
      # do the same for kappa2, theta, rhox
      kappa2 <- kappa2[(n_buffer + 1):(rows - n_buffer), 
                       (n_buffer + 1):(columns - n_buffer)]
      theta <- theta[(n_buffer + 1):(rows - n_buffer), 
                     (n_buffer + 1):(columns - n_buffer)]
      rhox <- rhox[(n_buffer + 1):(rows - n_buffer), 
                   (n_buffer + 1):(columns - n_buffer)]
      
      if (sanity_plotting == TRUE){
        par(mfrow = c(2,2))
        # first field
        plotgridList <- list( x= seq( 1,rows - (2*n_buffer),length.out= rows - (2*n_buffer)),
                              y= seq( 1,columns - (2*n_buffer),length.out= columns - (2*n_buffer)) )
        plotGrid<- make.surface.grid(plotgridList)
        image.plot(as.surface(plotGrid, f[,1]), col = turbo(256), main = "First field (trimmed)")
        
        # params
        image.plot(kappa2, col = viridis(256), main = "Kappa2 field (trimmed)")
        image.plot(theta, col = viridis(256), main = "Theta field (trimmed)")
        image.plot(rhox^2, col = viridis(256), main = "Rho field (trimmed)")
        par(mfrow = c(1,1))
      }
      
      # turn into vectors to combine with the spatial fields 
      kappa2 <- c(kappa2)
      theta <- c(theta)
      rho <- c(rhox^2)
      
      # combine all into one array 
      # we don't save rhoy or rhox, they come from rho
      final_sim <- cbind(f, kappa2, theta, rho)
      rm(f, kappa2, theta, rho)
      # print(dim(final_sim))
      dataset[,,sim] <- final_sim
      rm(final_sim)
    } # simulation (outer) loop
  ) # timing
  
  print(paste("Simulation took ", script_time[[3]]/60," minutes to run."))
  return(dataset)
  
} #function



############################################
## Experimental Setup and Hyperparameters ##
############################################

configs <- c("constant", "taper", "Gaussian", "Gaussian", 
             "coastline", "coastline", "sinwave", "double_Gaussian",
             "double_Gaussian", "double_coastline","double_coastline",
             "GP_gp", "GP_gp", "GP_gp", "GP_gp")

# Many different options for awght prior

# n_awghts <- 400
# awghts <- 4 + exp(seq(log(0.0001),
#                       log(2),
#                       length.out=n_awghts))
# awghts <- 4 + seq(sqrt(0.001), sqrt(2), length.out=n_awghts)^2
# awghts <- 4 + seq(0.0001, 2, length.out=n_awghts)

# Current awght prior
n_awghts_log <- 600      #100
n_awght_unif <- 400      #900
n_awghts <- n_awghts_log + n_awght_unif

awghts_log <- 4 + exp(seq(log(0.0001),
                          log(2),
                          length.out=n_awghts_log))
awghts_unif <- 4 + seq(0.0001, 2, length.out=n_awght_unif)

awghts <- c(awghts_log, awghts_unif)

low_awghts <- awghts[1:(n_awghts/2)]
high_awghts <- awghts[(n_awghts/2 + 1):n_awghts]

#sanity checks
summary(awghts)
summary(low_awghts)
summary(high_awghts)
summary(log(awghts-4))

# plotting for sanity 
par(mfrow = c(1,2))
hist(log(awghts-4), main = "log kappas", col = "lightgreen")
hist(awghts, main = "awghts", col = "gold")
par(mfrow = c(1,1))

rows <- 288
columns <- 192
n_buffer <- 10
random_seed <- 777

n_replicates <- 30
verbose <- FALSE 
sanity_plotting <- FALSE



###########################################
########## Simulation and Saving ##########
###########################################

nx <- rows * columns
ny <- n_replicates + 3

file_name <- here("data", "I2I_data.h5")
# file_name <- here("data", "I2I_sample_data.h5")
dataset_name <- "fields"

# create the file if it does not yet exist 
if (!file.exists(file_name)) {
  h5createFile(file_name)
  h5createDataset(
    file    = file_name,
    dataset = dataset_name,
    dims    = c(nx, ny, 0),              
    maxdims = c(nx, ny, H5Sunlimited()), 
    level   = 9,
    shuffle = FALSE
  )
}

for (start in seq(1, total_sims, by = chunk_size)) {
  this_chunk <- min(chunk_size, total_sims - start + 1)
  
  # Generate the data chunk
  generate_time <- system.time(
    dataset <- generate_i2i_data(
      N_SIMS = this_chunk, 
      n_replicates = n_replicates, 
      n_buffer = n_buffer,
      random_seed = random_seed, 
      configs = configs,
      awghts = awghts, 
      low_awghts = low_awghts, 
      high_awghts = high_awghts,
      rows = rows,
      columns = columns,
      verbose = verbose, 
      sanity_plotting = sanity_plotting
    )
  )
  
  print(paste("Generating data chunk took ", generate_time[[3]]/60," minutes."))
  
  dims <- get_dims()
  nz_old <- dims["nz"]
  nz_new <- this_chunk
  
  h5set_extent(
    file = file_name,
    dataset = dataset_name,
    dims = c(dims["nx"], dims["ny"], nz_old + nz_new)
  )
  
  write_time <- system.time(
    h5write(
      obj = dataset,
      file = file_name, 
      name = dataset_name, 
      index = list(1:nx, 1:ny, (nz_old + 1):(nz_old + nz_new))
    )
  )
  print(paste("Writing data chunk to h5 took ", write_time[[3]]/60," minutes."))
  h5closeAll()
  
  # for memory purposes 
  rm(dataset)
}


###########################################
### Loading in Subset of Data (Sanity) ####
###########################################
# 
# file_name <- here("data", "I2I_data.h5")
# dataset_name <- "fields"
# 
# print(h5ls(file_name))
# 
# rows <- 288
# columns <- 192
# nx <- rows * columns
# ny <- 33
# #choose how many simulations
# nz <- 4
# 
# subset_data <- h5read(file_name, dataset_name, index = list(1:nx, 1:ny, 1:nz))
# print(dim(subset_data))
# 
# 
# gridList<- list( x= seq( 1,rows,length.out= rows),
#                  y= seq( 1,columns,length.out= columns) )
# sGrid<- make.surface.grid(gridList)
# 
# 
# for (i in 1:4){
#   k = i
# par(mfrow = c(2,3))
# for (i in c(1,30,31,32,33)){
#   if (i == 1 || i == 30){
#     image.plot(as.surface(sGrid, subset_data[,i,k]), col = turbo(256) )
#   } else {
#     image.plot(as.surface(sGrid, subset_data[,i,k]), col = viridis(256))
#   }
# }
# par(mfrow = c(1,1))
# }
##########             IMPORTANT              ###########
#-------------------------------------------------------#
# Navigate to "Session" in the top left of RStudio and
# click on "Set Working Directory" and then 
# choose "To Source File Location". 
# Also, make sure you have downloaded all required 
# packages from required_packages.R. 
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
N_SIMS <- 80000
# sidelength of the field (window size)
sidelen <- 25 
# Other options can be tweaked in the
# "Experimental Setup and Hyperparameters" section below. 


###########################################
####### Full Data Generator #######
###########################################

generate_cnn_data <- function(
    N_SIMS, 
    n_replicates = 30,
    n_buffer = 5,
    random_seed = 777,
    awghts, 
    sidelen = 25, 
    verbose = FALSE, 
    sanity_plotting = FALSE){
  
  dataset <- array(data = NA, dim = c( ( sidelen^2 ), n_replicates + 1, N_SIMS))
  print("Dataset dimensions:") 
  print(dim(dataset))
  
  # incorporate buffer (will chop off later)
  sidelen <- sidelen + (2*n_buffer)
  n <- sidelen^2
  
  #grid for data 
  gridList<- list( x= seq( 1,sidelen,length.out= sidelen),
                   y= seq( 1,sidelen,length.out= sidelen) )
  sGrid<- make.surface.grid(gridList)
  
  script_time <- system.time(
    for (sim in 1:N_SIMS){
      # set.seed(random_seed + sim)
      
      # choose params 
      kappa2 <- sample(awghts, 1) - 4
      theta <- runif(1, -pi/2, pi/2)
      rho <- runif(1, 1, 7)
      
      rhox <- sqrt(rho)
      rhoy <- 1/rhox
      
      if (sim %% 100 == 0){
        print(paste("Loop #", sim))
      }
      
      if (verbose == TRUE){
        print(paste("kappa2:", kappa2, "rho:", rho, "theta:", theta))
      }
      
      # populate H matrix 
      H11 <- ( rhox^2 * (cos(theta))^2) + ( rhoy^2 * (sin(theta))^2 ) 
      H12 <- (rhoy^2 - rhox^2)*(sin(theta)*cos(theta))
      H21 <- H12 
      H22 <- (rhox^2 * (sin(theta))^2) + (rhoy^2 * (cos(theta))^2)
      
      factor <- 0.5
      SAR_stencil <- c( rbind( c(factor*H12, -H22, -factor*H12),
                                  c(-H11, kappa2 + 2*H11 + 2*H22, -H11),
                                  c(-factor*H12, -H22, factor*H12) ) )
      
      awght_obj <- list( values=SAR_stencil)
      class(awght_obj )<- "constantValue"
      
      LKinfo <- LKrigSetup(sGrid, NC =sidelen, 
                           nlevel = 1,
                           a.wghtObject =  awght_obj, 
                           normalize=FALSE, 
                           NC.buffer = 0, overlap = 2.5)
      
      f <- LKrig.sim( sGrid, LKinfo, M = n_replicates, just.coefficients = TRUE)
      mu <- rowMeans(f)
      sd <- apply(f, 1, sd)
      
      #normalization 
      f <- (f - mu)/sd
      
      # trim off buffer 
      f_trim <- array( f, dim = c(sidelen, sidelen, 30))
      f_trim <- f_trim[(n_buffer + 1):(sidelen - n_buffer), 
                   (n_buffer + 1):(sidelen - n_buffer), ]
      f_trim <- array(f_trim, dim = c( (sidelen - (2*n_buffer))*(sidelen - (2*n_buffer)), n_replicates))
      
      
      if (sanity_plotting == TRUE){
        par(mfrow = c(1,2))
        
        plotgridList <- list( x= seq( 1,sidelen - (2*n_buffer),length.out= sidelen - (2*n_buffer)),
                              y= seq( 1,sidelen - (2*n_buffer),length.out= sidelen - (2*n_buffer)) )
        plotGrid<- make.surface.grid(plotgridList)
        
        image.plot( as.surface( sGrid, f[,1]) , col = turbo(256), 
                    main = "First field")
        image.plot( as.surface( plotGrid, f_trim[,1]) , col = turbo(256), 
                    main = "First field (trimmed)")
      }
      
      # add vector with params (the remaining entries are zeroes)
      final_sim <- cbind(f_trim, rep(0, times = nrow(f_trim)))
      
      final_sim[1,(n_replicates + 1)] <- kappa2
      final_sim[2,(n_replicates + 1)] <- theta
      final_sim[3,(n_replicates + 1)] <- rho
      
      if (verbose == TRUE){
        print("Sim + param dims:")
        print(dim(final_sim))
      }
      
      dataset[,,sim] <- final_sim
      
    } # simulation for loop
  ) # timing 
  
  print(paste("Simulation took ", script_time[[3]]/60," minutes to run."))
  return(dataset)
  
} # function 


############################################
## Experimental Setup and Hyperparameters ##
############################################

n_awghts_log <- 600
n_awght_unif <- 400
# awghts <- 4 + seq(sqrt(0.001), sqrt(2), length.out=n_awghts)^2
# awghts <- 4 + seq(0.0001, 2, length.out=n_awghts)
awghts_log <- 4 + exp(seq(log(0.0001),
                      log(2),
                      length.out=n_awghts_log))
awghts_unif <- 4 + seq(0.0001, 2, length.out=n_awght_unif)

awghts <- c(awghts_log, awghts_unif)

#sanity checks
summary(awghts)

# sanity plotting
par(mfrow = c(1,2))
hist(log(awghts-4), main = "log kappas", col = "lightgreen")
hist(awghts, main = "awghts", col = "gold")
par(mfrow = c(1,1))


n_replicates <- 30 
n_buffer <- 5
verbose <- FALSE
sanity_plotting <- FALSE
random_seed <- 777


###########################################
########## Simulation and Saving ##########
###########################################

# Simulation
dataset <- generate_cnn_data(
  N_SIMS = N_SIMS, 
  n_replicates = n_replicates,
  n_buffer = n_buffer,
  random_seed = random_seed,
  awghts = awghts, 
  sidelen = sidelen, 
  verbose = verbose, 
  sanity_plotting = sanity_plotting
)

# Saving the dataset 

file_name <- here("data", "CNN_data.h5")
#file_name <- here("data", "CNN_sample_data.h5")
dataset_name <- "fields"

nx <- sidelen^2
ny <- n_replicates + 1
nz <- N_SIMS

h5createFile(file_name)

h5createDataset(
  file = file_name, 
  dataset = dataset_name,
  dims = c(nx,ny,nz),
  maxdims = c(nx,ny, H5Sunlimited()),
  level = 9, 
  shuffle = FALSE
)

# time how long it takes to write 
write_time <- system.time(
  h5write(
    obj = dataset,
    file = file_name, 
    name = dataset_name, 
    index = list(1:nx, 1:ny, 1:nz)
  )
)
print(paste("Writing data to h5 took ", write_time[[3]]/60," minutes."))
# remember to close it
h5closeAll()


############################################
#### Loading in the Data (Sanity) ####
############################################

# file_name <- here("data", "CNN_data.h5")
# #file_name <- here("data", "CNN_sample_data.h5")
# dataset_name <- "fields"
# 
# print(h5ls(file_name))
# 
# dataset <- h5read(file_name, dataset_name)
# print(dim(dataset))
# 
# k <- 1
# replicate_num <- 7
# 
# gridList<- list( x= seq( 1,sidelen,length.out= sidelen),
#                  y= seq( 1,sidelen,length.out= sidelen) )
# sGrid<- make.surface.grid(gridList)
# 
# image.plot( as.surface( sGrid, dataset[,replicate_num,k]) , col = turbo(256),
#             main = "Sample Field",
#             xlab = paste("Kappa2:", round(dataset[1, n_replicates+1, k], 4),
#                          "Theta:", round(dataset[2, n_replicates+1, k], 3),
#                          "Rho:", round(dataset[3, n_replicates+1, k], 3))
#             )

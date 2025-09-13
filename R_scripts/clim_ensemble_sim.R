##########             IMPORTANT              ###########
#-------------------------------------------------------#
# Navigate to "Session" in the top left of RStudio and
# click on "Set Working Directory" and then 
# choose "To Source File Location". 
#-------------------------------------------------------#

library(LatticeKrig)
library(spam64)
library(tictoc)
library(rhdf5)
library(maps)
library(cmocean)
library(here)
# helper functions
source(here("R_scripts", "helper_funcs.R"))

# Please choose which Climate model you are working with: 
Climate_model <- "EC3" #can be "CESM", "MPI", "EC3"


###########################################
######   Data and Initial Setup    #####
###########################################

if (Climate_model == "CESM"){
  rows <- 288
  columns <- 192
} else if (Climate_model == "MPI"){
  rows <- 192
  columns <- 96
} else if (Climate_model == "EC3"){
  rows <- 512
  columns <- 256
} else{
  print("Please select a valid model form CESM, MPI, or EC3.")
}


# load in i2i network clim outputs
file_path_s <- here(
  "results",
  "clim_outputs",
  "TransUNet_30rep_CESM__temp_sens_output.h5"
)
h5ls(file_path_s)
df_STUN <- h5read(file_path_s, "clim_output")

# extract/make params
kappa2_s <- exp(df_STUN[,,1])
awght_s <- kappa2_s + 4

# need to shift for LK 
theta_s <- pi/2 - df_STUN[,,2]

# for plotting intuitive angles
# theta_s <- -df_STUN[,,2]

rho_s <- df_STUN[,,3]
rhox_s <- sqrt(rho_s)
rhoy_s <- 1/rhox_s

# load in CNN outputs
file_path_c <- here(
  "results",
  "clim_outputs",
  "CNN25_30rep_CESM__temp_sens_output.h5"
)
h5ls(file_path_c)
df_CNN <- h5read(file_path_c, "clim_output")

# extract/make params
kappa2_c <- exp(df_CNN[,,1])
awght_c <- kappa2_c + 4

# need to shift for LK 
theta_c <- pi/2 - df_CNN[,,2]

rho_c <- df_CNN[,,3]
rhox_c <- sqrt(rho_c)
rhoy_c <- 1/rhox_c

# Load in the Climate/temp data 
if (Climate_model == "CESM"){
  
  load(
    here(
      "data",
      "JJAPatternScalingSlope.rda"
    )
  )
  JJASlopeNorm <- JJASlope
  for (field in 1:dim(JJASlope)[3]){
    JJASlopeNorm[,,field] <- (JJASlopeNorm[,,field] - JJASlopeMean)/JJASlopeSd
  }
  
  SlopeNorm <- pacific_centering(JJASlopeNorm)
  JJASlope <- pacific_centering(JJASlope)
  JJASlopeMean <- pacific_centering(JJASlopeMean)
  JJASlopeSd <- pacific_centering(JJASlopeSd)
  
} else if (Climate_model == "MPI"){
  
  file_path_clim <- here(
    "data",
    "mpi_temp_sens_fields.h5"
  )
  h5ls(file_path_clim)
  SlopeNorm <- h5read(file_path_clim, "clim_fields_norm")
  
} else if (Climate_model == "EC3"){
  file_path_clim <- here(
    "data",
    "ec3_temp_sens_fields.h5"
  )
  h5ls(file_path_clim)
  SlopeNorm <- h5read(file_path_clim, "clim_fields_norm")
} else{
  print("Please select a valid model form CESM, MPI, or EC3.")
}

# Coordinates and data grids 
xcoord = c(1:dim(SlopeNorm)[1])
ycoord = c(1:dim(SlopeNorm)[2])
# lon <- seq(-180, 180, length.out = dim(SlopeNorm)[1])
lon <- seq(0,360, length.out = dim(SlopeNorm)[1])
lat <- seq(-90, 90, length.out = dim(SlopeNorm)[2])

gridList<- list( x= seq( 1,rows,length.out= rows),
                 y= seq( 1,columns,length.out= columns) )
sGrid<- make.surface.grid(gridList)


###########################################
######   Initial Visualizations   #####
###########################################

first_field_norm <- SlopeNorm[,,1]

imagePlot(x = lon, y = lat,
          first_field_norm, main = "first field", 
          col = turbo(256))
map("world2", add = TRUE, 
    col = "grey0", lwd = 1)

# plot stun params with field
plot_params(kappa2_s, theta_s, rho_s, border_lwd = 1)

# plot cnn params with field
plot_params(kappa2_c, theta_c, rho_c, border_lwd = 1)


###########################################
######   Generating Ensembles   #####
###########################################


# get synthetic replicates and associated lkinfos
object_s <- generate_synthetic_reps(
  kappa2 = kappa2_s,
  theta = theta_s,
  rho = rho_s,
  rhox = rhox_s,
  rhoy = rhoy_s,
  n_replicates = 1000,
  random_seed = 6777,
  smooth_choice = FALSE,
  normalize = TRUE
)

object_c <- generate_synthetic_reps(
  kappa2 = kappa2_c,
  theta = theta_c,
  rho = rho_c,
  rhox = rhox_c,
  rhoy = rhoy_c,
  n_replicates = 1000,
  random_seed = 6777,
  smooth_choice = FALSE,
  normalize = TRUE
)

# extract LKinfo objects
LKinfo_s <- object_s$LKinfo
LKinfo_c <- object_c$LKinfo


###########################################
#### Covariance Experiments and Tests   ###
###########################################

set.seed(777)
# number of random points, 50
n_points <- 50

# pre-allocate a data.frame to store locations + RMSEs
results <- data.frame(
  pointx    = integer(n_points),
  pointy    = integer(n_points),
  rmse_stun = numeric(n_points),
  rmse_cnn  = numeric(n_points)
)

# start timer
tic("Total computation")

for (i in seq_len(n_points)) {
  # sample a random location
  px <- sample(seq_len(rows), 1)
  py <- sample(seq_len(columns), 1)
  results$pointx[i] <- px
  results$pointy[i] <- py
  
  # base correlation field
  target_ts <- as.vector(SlopeNorm[px, py, ])
  cors      <- apply(SlopeNorm, c(1,2), function(x) cor(x, target_ts))
  
  # STUN correlation field
  f_stun          <- array(object_s$f, dim = c(rows, columns, 1000))
  target_ts_stun  <- as.vector(f_stun[px, py, ])
  cors_stun       <- apply(f_stun, c(1,2), function(x) cor(x, target_ts_stun))
  
  # CNN correlation field
  f_cnn           <- array(object_c$f, dim = c(rows, columns, 1000))
  target_ts_cnn   <- as.vector(f_cnn[px, py, ])
  cors_cnn        <- apply(f_cnn, c(1,2), function(x) cor(x, target_ts_cnn))
  
  # compute and store RMSEs
  results$rmse_stun[i] <- sqrt(mean((cors       - cors_stun) ^ 2))
  results$rmse_cnn[i]  <- sqrt(mean((cors       - cors_cnn)  ^ 2))
}

# stop timer and compute elapsed time
tt  <- toc(log = TRUE)
elapsed_time <- tt$toc - tt$tic

# sum‐total of all RMSE values
sum_total_rmse_stun <- sum(results$rmse_stun)
sum_total_rmse_cnn <- sum(results$rmse_cnn)

# print summaries
cat("Sum total RMSE i2i empirical: ", sum_total_rmse_stun, "\n")
cat("Sum total RMSE cnn empirical: ", sum_total_rmse_cnn, "\n")
cat("Total time (s): ", elapsed_time, "\n")


# not paired (shouldnt use)
t.test(results$rmse_stun, results$rmse_cnn, 
       paired = FALSE, 
       alternative = "less", 
       conf.level = 0.99)

# paired (should use)
obj <- t.test(results$rmse_stun, results$rmse_cnn, 
              paired = TRUE, 
              alternative = "less", 
              conf.level = 0.99)
print(obj)


###########################################
###   Covariance Visualization (ENSO)   ###
###########################################

if (Climate_model == "CESM"){
  
  tic()
  # either randomly sample a point 
  pointx <- sample(1:rows,1)
  pointy <- sample(1:columns,1)
  # or use the Nino3.4 one (one of these pairs must be commented out)
  pointx <- 170 # enso choice 
  pointy <- 97
  print(c(pointx,pointy))
  
  # calculate correlations
  target_ts <- as.vector(SlopeNorm[pointx, pointy, ])  
  cors <- apply(SlopeNorm, c(1, 2), function(x) cor(x, target_ts))
  
  f_stun <- array(object_s$f, dim = c(rows, columns, 1000))
  target_ts_stun <- as.vector(f_stun[pointx, pointy, ])
  cors_stun <- apply(f_stun, c(1, 2), function(x) cor(x, target_ts_stun))
  
  f_cnn <- array(object_c$f, dim = c(rows, columns, 1000))
  target_ts_cnn <- as.vector(f_cnn[pointx, pointy, ])
  cors_cnn <- apply(f_cnn, c(1, 2), function(x) cor(x, target_ts_cnn))
  
  # take the rmse
  sqrt(mean((cors - cors_stun)^2 ))
  sqrt(mean((cors - cors_cnn)^2 ))
  toc()
  
  
  #trim poles for visuals
  vertchop <- 11:182
  latv     <- lat[vertchop]
  
  
  # compute zlims for covs
  zmax <- max(cors, cors_stun, cors_cnn)
  zmin <- min(cors, cors_stun, cors_cnn)
  zlim = c(-zmax, zmax)
  
  colorchoice = cmocean("balance")(22)
  bordercol = "grey10"
  
  
  # compute common zlims for actual fields
  zmaxclim  <- max(SlopeNorm[,vertchop,1], f_stun[,vertchop,4], f_cnn[,vertchop,4])
  zminclim  <- min(SlopeNorm[,vertchop,1], f_stun[,vertchop,4], f_cnn[,vertchop,4])
  zlimclim  <- c(zminclim, zmaxclim)
  
  
  
  ### ACTUAL FIELDS
  # png("clim_compare_stun_cnn25.png", width=4000, height=1000, res=600)
  par(mfrow=c(1,3), mar=c(1,1,1,1), oma=c(1,1,0,0))
  
  # 1) true field
  image(
    x    = lon, y = latv, z = SlopeNorm[,vertchop,1],
    col  = turbo(256), zlim = zlimclim,
    xaxt = "n", xlab = "", ylab = ""
  )
  map("world2", add=TRUE, col=bordercol, lwd=0.5)
  
  # 2) STUN
  image(
    x    = lon, y = latv, z = f_stun[,vertchop,4],
    col  = turbo(256), zlim = zlimclim,
    xaxt = "n", yaxt = "n", xlab = "", ylab = ""
  )
  map("world2", add=TRUE, col=bordercol, lwd=0.5)
  
  # 3) CNN
  image(
    x    = lon, y = latv, z = f_cnn[,vertchop,4],
    col  = turbo(256), zlim = zlimclim,
    xaxt = "n", yaxt = "n", xlab = "", ylab = ""
  )
  map("world2", add=TRUE, col=bordercol, lwd=0.5)
  par(mfrow = c(1,1))
  # dev.off()
  
  
  # CORRELATIONS
  # png("cov_compare_stun_cnn25.png", width=4000, height=1000, res=600)
  par(mfrow=c(1,3), mar=c(1,1,1,1), oma=c(1,1,0,0))
  
  
  # 1) actual cov
  image(x = lon, y = latv,z = cors[,vertchop], col = colorchoice, main = "",
        zlim = zlim, 
        xlab = "", ylab = "")
  map("world2", add = TRUE, col = bordercol, lwd = 0.5)
  # 2) stun cov
  image(x = lon, y = latv,z =cors_stun[,vertchop], col = colorchoice, 
        main = "",
        zlim = zlim, yaxt = "n", 
        xlab = "", ylab = "")
  map("world2", add = TRUE, col = bordercol, lwd = 0.5)
  # 3) cnn cov
  image(x = lon, y = latv,z =cors_cnn[,vertchop], col = colorchoice, 
        main = "",
        zlim = zlim, yaxt = "n", 
        xlab = "", ylab = "")
  map("world2", add = TRUE, col = bordercol, lwd = 0.5)
  par(mfrow = c(1,1))
  # dev.off()
  
  
  # COLORBAR FOR ACTUAL FIELDS
  # png("clim_colorbar_v3_div.png", width = 2333, height = 2000, res = 600)
  par( mar=c(1,1,1,1))
  
  # will trim this plot
  image.plot(x = lon, y = latv,z =cors_cnn[,vertchop], col = colorchoice, 
             main = "",
             zlim = zlim, yaxt = "n", 
             xlab = "", ylab = "")
  # dev.off()
  
  # COLORBAR FOR ACTUAL COVS
  # png("cov_colorbar_v3_div.png", width = 2333, height = 2000, res = 600)
  par( mar=c(1,1,1,1))
  
  # will trim this plot
  image.plot(x = lon, y = latv,z =SlopeNorm[,vertchop,1], col = turbo(256), 
             main = "",
             zlim = zlimclim, yaxt = "n", 
             xlab = "", ylab = "")
  # dev.off()
} else{
  print("The example visualization is for CESM")
}

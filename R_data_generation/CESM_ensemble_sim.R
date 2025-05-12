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

#helper functions (including generation function)
source(
  "C:/Users/anton/Desktop/Research/Paper_3/LatticeVision/R_data_generation/helper_funcs.R"
)

###########################################
######   Data and Initial Setup    #####
###########################################
rows <- 288
columns <- 192


# load in Vision Network (STUN,I2I,etc) outputs -------------
file_path_s <- "data/modelTransUNet_reps30_posRotaryPosEmbed_clim_output.h5"
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

# load in CNN outputs ---------------------
file_path_c <- "data/modelCNN_size25_reps30_clim_output.h5"
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

# Load in the Climate/temp data -----------------
load("C:/Users/anton/Desktop/Research/Paper_3/LatticeVision/sample_data/JJAPatternScalingSlope.rda")

# normalize all clim fields 
JJASlopeNorm <- JJASlope
for (field in 1:dim(JJASlope)[3]){
  JJASlopeNorm[,,field] <- (JJASlopeNorm[,,field] - JJASlopeMean)/JJASlopeSd
}

JJASlopeNorm <- pacific_centering(JJASlopeNorm)
JJASlope <- pacific_centering(JJASlope)
JJASlopeMean <- pacific_centering(JJASlopeMean)
JJASlopeSd <- pacific_centering(JJASlopeSd)


# Coordinates and data grids -------------
xcoord = c(1:dim(JJASlope)[1])
ycoord = c(1:dim(JJASlope)[2])
# lon <- seq(-180, 180, length.out = dim(JJASlope)[1])
lon <- seq(0,360, length.out = dim(JJASlope)[1])
lat <- seq(-90, 90, length.out = dim(JJASlope)[2])


gridList<- list( x= seq( 1,rows,length.out= rows),
                 y= seq( 1,columns,length.out= columns) )
sGrid<- make.surface.grid(gridList)


###########################################
######   Initial Visualizations   #####
###########################################


first_field <- JJASlope[,,1] 
first_field_norm <- JJASlopeNorm[,,1]

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
  rmse_cov  = numeric(n_points),
  rmse_stun = numeric(n_points),
  rmse_c    = numeric(n_points),
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
  
  # compute normalized covariance surfaces
  cov_c <- plot_cov_surface(
    LKinfo     = LKinfo_c,
    real_field = first_field_norm,
    loc_x      = px, 
    loc_y      = py,
    plotting   = FALSE
  )
  cor_c <- cov_c / max(cov_c)
  
  cov_s <- plot_cov_surface(
    LKinfo     = LKinfo_s,
    real_field = first_field_norm,
    loc_x      = px, 
    loc_y      = py,
    plotting   = FALSE
  )
  cor_s <- cov_s / max(cov_s)
  
  # base correlation field
  target_ts <- as.vector(JJASlopeNorm[px, py, ])
  cors      <- apply(JJASlopeNorm, c(1,2), function(x) cor(x, target_ts))
  
  # STUN correlation field
  f_stun          <- array(object_s$f, dim = c(rows, columns, 1000))
  target_ts_stun  <- as.vector(f_stun[px, py, ])
  cors_stun       <- apply(f_stun, c(1,2), function(x) cor(x, target_ts_stun))
  
  # CNN correlation field
  f_cnn           <- array(object_c$f, dim = c(rows, columns, 1000))
  target_ts_cnn   <- as.vector(f_cnn[px, py, ])
  cors_cnn        <- apply(f_cnn, c(1,2), function(x) cor(x, target_ts_cnn))
  
  # compute and store RMSEs
  results$rmse_cov[i]  <- sqrt(mean((cors       - cor_s)     ^ 2))
  results$rmse_stun[i] <- sqrt(mean((cors       - cors_stun) ^ 2))
  results$rmse_c[i]    <- sqrt(mean((cors       - cor_c)     ^ 2))
  results$rmse_cnn[i]  <- sqrt(mean((cors       - cors_cnn)  ^ 2))
}

# stop timer and compute elapsed time
tt  <- toc(log = TRUE)
elapsed_time <- tt$toc - tt$tic

# sum‐total of all RMSE values
sum_total_rmse_s <- sum(results$rmse_cov)
sum_total_rmse_stun <- sum(results$rmse_stun)
sum_total_rmse_c <- sum(results$rmse_c)
sum_total_rmse_cnn <- sum(results$rmse_cnn)


# print summaries
cat("Sum total RMSE stun exact: ", sum_total_rmse_s, "\n")
cat("Sum total RMSE stun empirical: ", sum_total_rmse_stun, "\n")
cat("Sum total RMSE cnn exact: ", sum_total_rmse_c, "\n")
cat("Sum total RMSE cnn empirical: ", sum_total_rmse_cnn, "\n")
cat("Total time (s): ", elapsed_time, "\n")


t.test(results$rmse_stun, results$rmse_cnn)
t.test(results$rmse_stun, results$rmse_cnn, paired = TRUE)



###########################################
###   Covariance Visualization (ENSO)   ###
###########################################


tic()
pointx <- sample(1:rows,1)
pointy <- sample(1:columns,1)
pointx <- 170 # enso choice 
pointy <- 97
print(c(pointx,pointy))

cov_c <- plot_cov_surface(
  LKinfo = LKinfo_c,
  real_field = first_field_norm,
  loc_x = pointx, 
  loc_y = pointy,
  plotting = FALSE
)
cor_c <- cov_c/max(cov_c)

cov_s <- plot_cov_surface(
  LKinfo = LKinfo_s,
  real_field = first_field_norm,
  loc_x = pointx, 
  loc_y = pointy,
  plotting = FALSE
)
cor_s <- cov_s/max(cov_s)
target_ts <- as.vector(JJASlopeNorm[pointx, pointy, ])  
cors <- apply(JJASlopeNorm, c(1, 2), function(x) cor(x, target_ts))
f_stun <- array(object_s$f, dim = c(rows, columns, 1000))
target_ts_stun <- as.vector(f_stun[pointx, pointy, ])
cors_stun <- apply(f_stun, c(1, 2), function(x) cor(x, target_ts_stun))
f_cnn <- array(object_c$f, dim = c(rows, columns, 1000))
target_ts_cnn <- as.vector(f_cnn[pointx, pointy, ])
cors_cnn <- apply(f_cnn, c(1, 2), function(x) cor(x, target_ts_cnn))

# take the rmse of cors and cov_surf_s
sqrt(mean((cors - cor_s)^2 ))
sqrt(mean((cors - cors_stun)^2 ))
sqrt(mean((cors - cor_c)^2 ))
sqrt(mean((cors - cors_cnn)^2 ))
toc()


#trim poles
vertchop <- 11:182
latv     <- lat[vertchop]

# compute zlims for covs
zmax <- max(cors, cors_stun, cors_cnn)
zmin <- min(cors, cors_stun, cors_cnn)
zlim = c(-zmax, zmax)

colorchoice = cmocean("balance")(22)

bordercol = "grey10"


# compute common z‐limits for actual fields
zmaxclim  <- max(JJASlopeNorm[,vertchop,1], f_stun[,vertchop,4], f_cnn[,vertchop,4])
zminclim  <- min(JJASlopeNorm[,vertchop,1], f_stun[,vertchop,4], f_cnn[,vertchop,4])
zlimclim  <- c(zminclim, zmaxclim)



### ACTUAL FIELDS
# png("clim_compare_vit_cnn9.png", width=4000, height=1000, res=600)
par(mfrow=c(1,3), mar=c(1,1,1,1), oma=c(1,1,0,0))

# 1) true field
image(
  x    = lon, y = latv, z = JJASlopeNorm[,vertchop,1],
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
# png("cov_compare_vit_cnn9.png", width=4000, height=1000, res=600)
par(mfrow=c(1,3), mar=c(1,1,1,1), oma=c(1,1,0,0))


image(x = lon, y = latv,z = cors[,vertchop], col = colorchoice, main = "",
      zlim = zlim, 
      xlab = "", ylab = "")
map("world2", add = TRUE, col = bordercol, lwd = 0.5)
#stun cov
image(x = lon, y = latv,z =cors_stun[,vertchop], col = colorchoice, 
      main = "",
      zlim = zlim, yaxt = "n", 
      xlab = "", ylab = "")
map("world2", add = TRUE, col = bordercol, lwd = 0.5)
#cnn cov
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

image.plot(x = lon, y = latv,z =cors_cnn[,vertchop], col = colorchoice, 
           main = "",
           zlim = zlim, yaxt = "n", 
           xlab = "", ylab = "")
# dev.off()

# COLORBAR FOR ACTUAL COVS
# png("cov_colorbar_v3_div.png", width = 2333, height = 2000, res = 600)
par( mar=c(1,1,1,1))

image.plot(x = lon, y = latv,z =JJASlopeNorm[,vertchop,1], col = turbo(256), 
           main = "",
           zlim = zlimclim, yaxt = "n", 
           xlab = "", ylab = "")
# dev.off()


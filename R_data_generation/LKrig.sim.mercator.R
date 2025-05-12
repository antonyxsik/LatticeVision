library(LatticeKrig)
library(spam64)

rm(list=ls())
setwd("C:/Users/anton/Desktop/Research/Paper_3/LatticeVision")


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
# simulation function. here we make a slight edit
# to make the B matrix have periodic properties
# this allows us to respect the Mercator projection

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

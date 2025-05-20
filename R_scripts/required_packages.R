install.packages("LatticeKrig")
install.packages("spam64")
install.packages("tictoc")
install.packages("maps")
install.packages("cmocean")
install.packages("here")
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install(
  "rhdf5",
  ask    = FALSE,    
  update = FALSE 
)
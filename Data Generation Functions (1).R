################ CONTEST - MASL A.A. 2022-2023 ################
###############      (DGP FUNCTIONS)      #################
##### Gabriella Stabile, Chiara Cavigli, Giuseppe Martinelli #####


# ************************* (N>P) ******************************
# (Dati che non presentano alta correlazione)
rm(list=ls())
dgp_Np <- function(seed=999, n=100, p=10){
  
  X <- matrix(nrow = n, ncol = p) # empty matrix
  for (i in 1:p) {                      # loop for columns
    set.seed(seed+i)                    # change seed at each iteration
    mu <- as.integer(runif(1,min=-3,max=5)) # generate mu
    sigma <- as.integer(runif(1,min=1,max=5)) # generate sigma
    colonna <- matrix(rnorm(n=n, mu ,sigma)) # generate n units for each column
    X[,i] <- colonna                        # add column to matrix
  }
  
  beta <- matrix(as.integer(runif(p,min=-3,max=3))) # generate coefficients
  Y <- -5 + X %*% beta + rnorm(n,0, 1) # simulate Y
  simulated <- data.frame(cbind(Y,X)) # combine X and Y in one dataframe
  colnames(simulated) <- c("Y", c(paste0("X", 1:(NCOL(simulated)-1) ))) # rename
  return(simulated)}

# Correlazione nei Dati
#dati1 <- dgp_Np()   
#X <- dati1[,-1]  
#round(cor(X),2)
#round(ggm::parcor(var(X)),2)


# ************************* (P>N) ******************************
# (Dati che non presentano alta correlazione)
dgp_Pn <- function(seed=999, n=100, p=200){
  
  X <- matrix(nrow = n, ncol = p) # empty matrix
  for (i in 1:p) {                      # loop for columns
    set.seed(seed+i)                    # change seed at each iteration
    mu <- as.integer(runif(1,min=-3,max=5)) # generate mu
    sigma <- as.integer(runif(1,min=1,max=5)) # generate sigma
    colonna <- matrix(rnorm(n=n, mu ,sigma)) # generate n units for each column
    X[,i] <- colonna                        # add column to matrix
  }
  
  beta <- matrix(as.integer(runif(p,min=-3,max=3))) # generate coefficients
  Y <- -5 + X %*% beta + rnorm(n,0, 1) # simulate Y
  simulated <- data.frame(cbind(Y,X)) # combine X and Y in one dataframe
  colnames(simulated) <- c("Y", c(paste0("X", 1:(NCOL(simulated)-1) ))) # rename
  return(simulated)}

# Cor, Parcor non funzionano in questo caso.


# ************************* (N>P) ******************************
# (Dati che presentano alcuni predittori perfettamente correlati)

dgp_Np_coll <- function(seed=999, n=100, p=10, signif=8, colli=2){
  
  X <- matrix(nrow = n, ncol = p)        # empty matrix
  for (i in 1:signif) {                  # loop for columns
    set.seed(seed+i)                     # change seed at each iteration
    mu <- as.integer(runif(1,min=-3,max=5)) # generate mu
    sigma <- as.integer(runif(1,min=1,max=5)) # generate sigma
    colonna <- matrix(rnorm(n=n, mu ,sigma)) # generate n units for each column
    X[,i] <- colonna                        # add column to matrix
  }
  
  # Generiamo manualmente due variabili collineari ad altre
  X[,9] <- X[,1] + 2*X[,2]
  X[,10] <- X[,1] - 5*X[,5]
  
  beta <- matrix(as.integer(runif(p,min=-3,max=3))) # generate coefficients
  Y <- -5 + X %*% beta + rnorm(n,0, 1) # simulate Y
  simulated <- data.frame(cbind(Y,X)) # combine X and Y in one dataframe
  colnames(simulated) <- c("Y", c(paste0("X", 1:(NCOL(simulated)-1) ))) # rename
  return(simulated)}

# Correlazione nei dati 
#dati1 <- dgp_Np_coll()
#X <- dati1[,-1]  
#round(cor(X),2)
#linear <- lm(Y~., data=dati1)
#linear$rank # rango non pieno

# ************************* (P>N) ******************************
# (Dati altamente correlati)

dgp_Pn_coll <- function(seed=999, n=100, p=200){
  set.seed(seed)
  sigma_err <- 1
  Corr_mat <- clusterGeneration::rcorrmatrix(p, alphad=1)
  Bvec <- as.integer(runif(n=p,min=-1,max=5))
  
  X <- mvtnorm::rmvnorm(n,sigma = Corr_mat)
  Y <- X %*% Bvec + rnorm(n,0,sigma_err)
  
  simulated <- data.frame(cbind(Y,X)) # combine X and Y in one dataframe
  colnames(simulated) <- c("Y", c(paste0("X", 1:(NCOL(simulated)-1) ))) # rename
  return(simulated)}

#dati1 <- dgp_Pn_coll()
#X <- dati1[,-1] 
#head(dati1)

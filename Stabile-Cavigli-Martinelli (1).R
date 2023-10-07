############### CONTEST - MASL A.A. 2022-2023 ################
###############           (MAIN FILE)         #################
##### Gabriella Stabile, Chiara Cavigli, Giuseppe Martinelli #####

# 1) ---------------------- (N>p) ------------------------
# Nel primo caso osserviamo come si comportano i modelli nel contesto n>p,
# con predittori generati indipendentemente (correlazione tra loro bassa).
# La Y è generata come combinazione lineare dei predittori e un vettore
# di beta interi, più un'intercetta e un termine di errore (gaussiano).
# Sia i predittori che la variabile di risposta sono continue/numeriche.
# Ci si aspetta che il modello lineare faccia un ottimo lavoro.

rm(list=ls())
source("/Users/gab/Uni/Magistrale/Multivariata/Contest MASL/Consegna Contest/Data Generation Functions.R")
library("glmnet")
library("ggm")

# Generiamo i dati
dati1 <- as.matrix(dgp_Np(n=200, seed=2022)); dim(dati1) # n=200, p=10
Y <- dati1[,1]    # Y column
X <- dati1[,-1]   # X columns

# Train, Validation, Test sets
campionamento <- sample(c(1, 2, 3), nrow(dati1), replace=TRUE, prob=c(0.7,0.15,0.15))
train <- dati1[campionamento==1,]     ; dim(train)       # n=143 p=10
validation <- dati1[campionamento==2,]; dim(validation ) # n=28 p=10
test <- dati1[campionamento==3,]      ; dim(test)        # n=29 p=10

# Correlazione e Correlazione parziale (relativamente basse)
round(cor(X),2)              
round(ggm::parcor(var(X)),2) 

# ================= (LINEARE) =====================
# Nota. Non usiamo il validation set qui.
lineare <- lm(Y~., data=data.frame(train)) # Training 
lineare$coefficients 
lineare$rank # rango 11, pieno
mean(lineare$residuals^2) # MSE train = 0.8544215
predicted_linear <- predict(lineare, newdata=data.frame(test)) # Testing
mean((predicted_linear-test[,1])^2) # MSE test= 1.140123

# =================== (RIDGE) =====================
# In questo caso (N>p senza alta correlazione tra i predittori) ci aspettiamo
# che la crossvalidation scelga un lambda piccolo.

# (choosing shrinkage parameter via cross validation)
cv_ridge <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_ridge_lambda <- cv_ridge$lambda.min # lambda = 0.81485, log(lambda)= -0.21

# (ridge model training)
ridge_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0)
range(log(ridge_mod$lambda)) # range log(lambda) = [-0.1255311, 9.0848093]
plot(ridge_mod, xvar="lambda",label=TRUE, main = "Ridge")
abline(v=log(cv_ridge$lambda.min), lty="dashed", col="red")
coef(ridge_mod, s=cv_ridge$lambda.min) # stime sul train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(ridge_mod, newx=train[,-1], s=cv_ridge_lambda)
mean((pred-train[,1])^2) # MSE train = 1.50594

# (stime sul test)
predicted_values_r <- predict(ridge_mod, newx=test[,-1], s=cv_ridge_lambda)
mean((predicted_values_r-test[,1])^2) # MSE = 2.066588

# ====================== (LASSO) ======================
# (crossvalidation & lambda that minimizes crossval error)
cv_lasso <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=1, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_lasso_lambda <- cv_lasso$lambda.min # lambda = 0.1028218

# (lasso model training)
lasso_mod <- glmnet(x=train[,-1], y=train[,1], alpha=1)
plot(lasso_mod, xvar="lambda",label=TRUE, main = "Lasso")
abline(v=log(cv_lasso$lambda.min), lty="dashed", col="red")
coef(lasso_mod, s=cv_lasso$lambda.min) # stime coef del train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(lasso_mod, newx=train[,-1], s=cv_lasso_lambda)
mean((pred-train[,1])^2) # MSE train = 0.9433884

# (stime sul test)
predicted_values_l <- predict(lasso_mod, newx=test[,-1], s=cv_lasso_lambda)
mean((predicted_values_l-test[,1])^2) # MSE test = 1.306006

# ====================== (ELASTIC) ======================
# (crossvalidation & lambda that minimizes crossval error)
cv_elastic <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0.5, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_elastic_lambda <- cv_elastic$lambda.min # lambda=0.1873747

# (elastic net model training)
elastic_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0.5)
plot(elastic_mod, xvar="lambda",label=TRUE, main = "Elastic")
abline(v=log(cv_elastic$lambda.min), lty="dashed", col="red")
coef(elastic_mod, s=cv_elastic$lambda.min) 

# (train MSE with crossvalidated tuning parameter)
pred <- predict(elastic_mod, newx=train[,-1], s=cv_elastic_lambda)
mean((pred-train[,1])^2) # MSE train = 0.985453

# (stime sul test)
predicted_values_e <- predict(elastic_mod, newx=test[,-1], s=cv_elastic_lambda)
mean((predicted_values_e-test[,1])^2) # MSE = 1.366887

# =================== (PRINCIPAL COMPONENT REGRESSION) ======================
library(pls)
dati1; X; Y; train; validation; test
set.seed (999)

# (validate for ideal number of components)
cv_pcr <- pcr(validation[,1]~., 
              data=data.frame(validation), 
              scale=TRUE, validation ="CV") 
summary(cv_pcr)
validationplot(cv_pcr,val.type="MSEP") # option MSEP plots MSE
# the model tells us all 10 components are relevant.

# (train)
pcr_mod <- pcr(train[,1]~., 
               data=data.frame(train), 
               scale=TRUE, ncomp=10) # RMSE
summary(pcr_mod)
validationplot(pcr_mod, val.type="MSEP")
pcr_mod$coefficients # coefficienti, beta componenti stimati sul train
mean((pcr_mod$residuals)^2) # MSE train= 1.989586

# (test)
predicted_pcr <- predict(pcr_mod, newdata=test, ncomp=10) 
predicted_pcr 
mean((predicted_pcr-test[,1])^2) # MSE test= 0.2657339

# ***************************************************************
# ****************************(N>p )***********************************
# ***************************************************************
# Con alcuni predittori (perfettamente) correlati

rm(list=ls())
source("/Users/gab/Uni/Magistrale/Multivariata/Contest MASL/Consegna Contest/Data Generation Functions.R")
library("glmnet")
library("ggm")

dati1 <- as.matrix(dgp_Np_coll(n=200)); dim(dati1)
Y <- dati1[,1]   
X <- dati1[,-1]   

# Train, Validation, Test sets
campionamento <- sample(c(1, 2, 3), nrow(dati1), replace=TRUE, prob=c(0.7,0.15,0.15))
train <- dati1[campionamento==1,]     ; dim(train)
validation <- dati1[campionamento==2,]; dim(validation )
test <- dati1[campionamento==3,]      ; dim(test)

#round(ggm::parcor(var(X)),2) # Non funziona per N > p collineare
#summary(dati1)

# ************************ (MODELLO LINEARE N>p collineare) *************************

# Vediamo che alcuni predittori hanno una relazione lineare perfetta.
pairs( dati1[,5:11], panel=function(x,y){
  points(x,y)
  abline(lm(y~x), col='red')
  text(0,1.5,labels = paste('R2=',round((cor(x,y))^2,2)) ,col='red' )
})

(train)
linear <- lm(Y~., data=data.frame(train))
linear$coefficients # Le covariate ridondanti X9 e X10 sono NA.
linear$rank # rango 9, non pieno
mean(linear$residuals^2) # MSE train = 0.9547065

# VIF
library(car)
vif(linear) # there are aliased coefficients in the model (c'è corr perfetta)

# (test)
predicted_linear <- predict(linear, newdata=data.frame(test)) 
predicted_linear 
mean((predicted_linear-test[,1])^2) # MSE= 1.086147
round(cor(X), 2)

# ************************ (RIDGE alpha=0) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_ridge <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_ridge_lambda <- cv_ridge$lambda.min # lambda = 0.8352604

# (ridge model training)
ridge_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0)
plot(ridge_mod, xvar="lambda",label=TRUE, main = "Ridge")
abline(v=log(cv_ridge$lambda.min), lty="dashed", col="red")
coef(ridge_mod, s=cv_ridge$lambda.min) # stime sul train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(ridge_mod, newx=train[,-1], s=cv_ridge_lambda)
mean((pred-train[,1])^2) # MSE train = 0.985453

# (stime sul test)
predicted_values_r <- predict(ridge_mod, newx=test[,-1], s=cv_ridge_lambda)
mean((predicted_values_r-test[,1])^2) # MSE = 1.649807

# ************************ (LASSO alpha=1) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_lasso <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=1, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_lasso_lambda <- cv_lasso$lambda.min # lambda = 0.04157111

# (lasso model training)
lasso_mod <- glmnet(x=train[,-1], y=train[,1], alpha=1)
plot(lasso_mod, xvar="lambda",label=TRUE, main = "Lasso")
abline(v=log(cv_lasso$lambda.min), lty="dashed", col="red")
coef(lasso_mod, s=cv_lasso$lambda.min) 

# (train MSE with crossvalidated tuning parameter)
pred <- predict(lasso_mod, newx=train[,-1], s=cv_lasso_lambda)
mean((pred-train[,1])^2) # MSE train = 0.9661396

# (stime sul test)
predicted_values_l <- predict(lasso_mod, newx=test[,-1], s=cv_lasso_lambda)
mean((predicted_values_l-test[,1])^2) # MSE = 1.079952


# ************************ (ELASTIC NET alpha=0.5) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_elastic <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0.5, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_elastic_lambda <- cv_elastic$lambda.min # lambda=0.04757703

# (elastic net model training)
elastic_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0.5)
plot(elastic_mod, xvar="lambda",label=TRUE, main = "Elastic")
abline(v=log(cv_elastic$lambda.min), lty="dashed", col="red")
coef(elastic_mod, s=cv_elastic$lambda.min) 

# (train MSE with crossvalidated tuning parameter)
pred <- predict(elastic_mod, newx=train[,-1], s=cv_elastic_lambda)
mean((pred-train[,1])^2) # MSE train = 0.9652501

# (stime sul test)
predicted_values_e <- predict(elastic_mod, newx=test[,-1], s=cv_elastic_lambda)
mean((predicted_values_e-test[,1])^2) # MSE = 1.078409


# ************************ (PCA REGRESSION) *************************
library(pls)
set.seed (999)

# validate for ideal number of components
cv_pcr <- pcr(validation[,1]~., 
              data=data.frame(validation), 
              scale=TRUE, validation ="CV") 
summary(cv_pcr)
validationplot(cv_pcr,val.type="MSEP") # 8 components

# train
pcr_mod <- pcr(train[,1]~., 
               data=data.frame(train), 
               scale=TRUE, ncomp=8) # train usando 8 componenti
summary(pcr_mod)
validationplot(pcr_mod, val.type="MSEP")
pcr_mod$coefficients 
mean(pcr_mod$residuals^2) # MSE train = 22.0642

# test
predicted_pcr <- predict(pcr_mod, newdata=test, ncomp=8) 
predicted_pcr 
mean((predicted_pcr-test[,1])^2) # MSE per ncomp=8, 0.2898464


# ***************************************************************
# ****************************(P>N)***********************************
# ***************************************************************
# Dati con bassa correlazione.

rm(list=ls())
source("/Users/gab/Uni/Magistrale/Multivariata/Contest MASL/Consegna Contest/Data Generation Functions.R")
library("glmnet")
library("ggm")

dati1 <- as.matrix(dgp_Pn(n=200, p=200)); dim(dati1)
Y <- dati1[,1]    
X <- dati1[,-1]   

# Train, Validation, Test sets
campionamento <- sample(c(1, 2, 3), nrow(dati1), replace=TRUE, prob=c(0.7,0.15,0.15))
train <- dati1[campionamento==1,]     ; dim(train) 
validation <- dati1[campionamento==2,]; dim(validation) 
test <- dati1[campionamento==3,]      ; dim(test) 


# ************************ (MODELLO LINEARE P>N) *************************
linear <- lm(Y~., data=data.frame(train))
linear$coefficients # molti NA
linear$rank # rango non pieno
mean(linear$residuals^2) # MSE train = 0. Overfitting.
predicted_linear <- predict(linear, newdata=data.frame(test)) 
predicted_linear 
mean((predicted_linear-test[,1])^2) # MSE test= 241221. 


# ************************ (RIDGE P>N) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_ridge <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_ridge_lambda <- cv_ridge$lambda.min # lambda = 253.6279

# (ridge model training)
ridge_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0)
plot(ridge_mod, xvar="lambda",label=TRUE, main = "Ridge")
abline(v=log(cv_ridge$lambda.min), lty="dashed", col="red")
coef(ridge_mod, s=cv_ridge$lambda.min) # stime sul train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(ridge_mod, newx=train[,-1], s=cv_ridge_lambda)
mean((pred-train[,1])^2) # MSE train = 1343.484

# (stime sul test)
predicted_values_r <- predict(ridge_mod, newx=test[,-1], s=cv_ridge_lambda)
mean((predicted_values_r-test[,1])^2) # MSE test = 2460.133

# ************************ (LASSO alpha=1) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_lasso <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=1, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_lasso_lambda <- cv_lasso$lambda.min # lambda = 25.36279

# (lasso model training)
lasso_mod <- glmnet(x=train[,-1], y=train[,1], alpha=1)
plot(lasso_mod, xvar="lambda",label=TRUE, main = "Lasso")
abline(v=log(cv_lasso$lambda.min), lty="dashed", col="red")
coef(lasso_mod, s=cv_lasso$lambda.min) # stime coef del train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(lasso_mod, newx=train[,-1], s=cv_lasso_lambda)
mean((pred-train[,1])^2) # MSE train = 3140.604

# (stime sul test)
predicted_values_l <- predict(lasso_mod, newx=test[,-1], s=cv_lasso_lambda)
mean((predicted_values_l-test[,1])^2) # MSE = 3002.814


# ************************ (ELASTIC NET alpha=0.5) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_elastic <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0.5, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_elastic_lambda <- cv_elastic$lambda.min  # lambda=50.72557

# (elastic net model training)
elastic_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0.5)
plot(elastic_mod, xvar="lambda",label=TRUE, main = "Elastic")
abline(v=log(cv_elastic$lambda.min), lty="dashed", col="red")
coef(elastic_mod, s=cv_elastic$lambda.min) # stime coef del train

pred <- predict(elastic_mod, newx=train[,-1], s=cv_elastic_lambda)
mean((pred-train[,1])^2) # MSE train = 48.30812

# (stime sul test)
predicted_values_e <- predict(elastic_mod, newx=test[,-1], s=cv_elastic_lambda)
mean((predicted_values_e-test[,1])^2) # MSE = 1496.969


# ************************ (PCA REGRESSION) *************************
library(pls)
set.seed (999)

# validate for ideal number of components
cv_pcr <- pcr(validation[,1]~., 
              data=data.frame(validation), 
              scale=TRUE, validation ="CV") # nota. Da il Root MSE
summary(cv_pcr)
validationplot(cv_pcr,val.type="MSEP") 

# train
pcr_mod <- pcr(train[,1]~., 
               data=data.frame(train), 
               scale=TRUE, ncomp=100) # RMSE
summary(pcr_mod)
validationplot(pcr_mod, val.type="MSEP") # 70 componenti
pcr_mod$coefficients # coefficienti, beta stimati sul train
mean(pcr_mod$residuals^2) # MSE train = 325.2073

# test
predicted_pcr <- predict(pcr_mod, newdata=test, ncomp=70) 
predicted_pcr # 30 Y fitted values
mean((predicted_pcr-test[,1])^2) # MSE test= 736.8714


# ***************************************************************
# ****************************(P>n alta corr)***********************************
# ***************************************************************

rm(list=ls())
source("/Users/gab/Uni/Magistrale/Multivariata/Contest MASL/Consegna Contest/Data Generation Functions.R")
library("glmnet")
library("ggm")

dati1 <- as.matrix(dgp_Pn_coll(n=150)); dim(dati1)
Y <- dati1[,1]   
X <- dati1[,-1]  

# Train, Validation, Test sets
campionamento <- sample(c(1, 2, 3), nrow(dati1), replace=TRUE, prob=c(0.7,0.15,0.15))
train <- dati1[campionamento==1,]     ; dim(train)
validation <- dati1[campionamento==2,]; dim(validation )
test <- dati1[campionamento==3,]      ; dim(test)

# ************************ (MODELLO LINEARE P>n alta corr) *************************

pairs( dati1[,27:29], panel=function(x,y){
  points(x,y)
  abline(lm(y~x), col='red')
  text(0,1.5,labels = paste('R2=',round((cor(x,y))^2,2)) ,col='red' )
})


linear <- lm(Y~., data=data.frame(train))
linear$coefficients 
linear$rank # rango 99, non pieno
mean(linear$residuals^2) # MSE train = 0


predicted_linear <- predict(linear, newdata=data.frame(test)) 
predicted_linear 
mean((predicted_linear-test[,1])^2) # MSE= 13822.87

# ************************ (RIDGE P>N alta corr *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_ridge <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_ridge_lambda <- cv_ridge$lambda.min # lambda = 188.2028

# (ridge model training)
ridge_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0)
plot(ridge_mod, xvar="lambda",label=TRUE, main = "Ridge")
abline(v=log(cv_ridge$lambda.min), lty="dashed", col="red")
coef(ridge_mod, s=cv_ridge$lambda.min) # stime sul train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(ridge_mod, newx=train[,-1], s=cv_ridge_lambda)
mean((pred-train[,1])^2) # MSE train = 420.4897

# (stime sul test)
predicted_values_r <- predict(ridge_mod, newx=test[,-1], s=cv_ridge_lambda)
mean((predicted_values_r-test[,1])^2) # MSE = 705.8466

# ************************ (LASSO alpha=1) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_lasso <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=1, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_lasso_lambda <- cv_lasso$lambda.min # lambda = 1.755186

# (lasso model training)
lasso_mod <- glmnet(x=train[,-1], y=train[,1], alpha=1)
plot(lasso_mod, xvar="lambda",label=TRUE, main = "Lasso")
abline(v=log(cv_lasso$lambda.min), lty="dashed", col="red")
coef(lasso_mod, s=cv_lasso$lambda.min) # stime coef del train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(lasso_mod, newx=train[,-1], s=cv_lasso_lambda)
mean((pred-train[,1])^2) # MSE train = 143.4278

# (stime sul test)
predicted_values_l <- predict(lasso_mod, newx=test[,-1], s=cv_lasso_lambda)
mean((predicted_values_l-test[,1])^2) # MSE = 640.0005


# ************************ (ELASTIC NET alpha=0.5) *************************
# (crossvalidation & lambda that minimizes crossval error)
cv_elastic <- cv.glmnet(x=validation[,-1], y=validation[,1], alpha=0.5, nfolds=10 ,family="gaussian",standardize=TRUE)
cv_elastic_lambda <- cv_elastic$lambda.min # lambda = 2.655467

# (elastic net model training)
elastic_mod <- glmnet(x=train[,-1], y=train[,1], alpha=0.5)
plot(elastic_mod, xvar="lambda",label=TRUE, main = "Elastic")
abline(v=log(cv_elastic$lambda.min), lty="dashed", col="red")
coef(elastic_mod, s=cv_elastic$lambda.min) # stime coef del train

# (train MSE with crossvalidated tuning parameter)
pred <- predict(elastic_mod, newx=train[,-1], s=cv_elastic_lambda)
mean((pred-train[,1])^2) # MSE train = 105.4596

# (stime sul test)
predicted_values_e <- predict(elastic_mod, newx=test[,-1], s=cv_elastic_lambda)
mean((predicted_values_e-test[,1])^2) # MSE = 585.8645


# ************************ (PCA REGRESSION) *************************
library(pls)
set.seed (999)

# (validate for ideal number of components)
cv_pcr <- pcr(validation[,1]~., 
              data=data.frame(validation), 
              scale=TRUE, validation ="CV") 
summary(cv_pcr)
validationplot(cv_pcr,val.type="MSEP") 

# (train)
pcr_mod <- pcr(train[,1]~., 
               data=data.frame(train), 
               scale=TRUE, ncomp=40) # 30 componenti 
summary(pcr_mod)
validationplot(pcr_mod, val.type="MSEP")
pcr_mod$coefficients 
mean(pcr_mod$residuals^2) # train MSE = 371.5847

# (test)
predicted_pcr <- predict(pcr_mod, newdata=test, ncomp=29) 
predicted_pcr 
mean((predicted_pcr-test[,1])^2) # MSE per ncomp=29, 505.5952







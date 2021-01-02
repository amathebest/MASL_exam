######## Ridge Regression and Lasso
library(glmnet)
library(ISLR)

# data import
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
data <- read.csv("housing.csv", header = F)
column_names = c('CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV')
colnames(data) <- column_names

head(data)

mod0 <- lm(MEDV ~ . -1, data = data)
summary(mod0)




# splitting the dataset into X and y
X = model.matrix(MEDV ~ . -1, data = data)
y = data$MEDV

# fitting the Ridge Regression
fit.ridge = glmnet(X, y, family = "gaussian", alpha = 0) # alpha=0 -> ridge. Default standardize=TRUE 

plot(fit.ridge, xvar = "lambda", label = TRUE)
# as log lambda increases, every coefficient converges to 0

dim(fit.ridge$beta)
round(fit.ridge$lambda[c(1,20,100)], 5)
round(fit.ridge$beta[,c(1,20,100)], 5)

# To choose lambda, us cross-val
### Model Selection by Cross-Validation
set.seed(123)
cv.ridge <- cv.glmnet(X, y, alpha = 0, nfolds = 10) 
# in each cross-validation, for every lambda compute the ridge estimator and predict the value of the response on the 1/10 of the data left (error estimate)
# at the end, for every lambda we have 10 values of RSS -> compute the mean among these values to complete the cross-validation
# from here we want the minimum value of lambda, which sometimes can also be not ideal
cv.ridge
plot(cv.ridge)

# in the plot there are 2 vertical lines for
cv.ridge$lambda.min
log(cv.ridge$lambda.min)
cv.ridge$lambda.1se
log(cv.ridge$lambda.1se)

# huge difference!
round(cbind(coef(cv.ridge, s=cv.ridge$lambda.min), coef(cv.ridge, s=cv.ridge$lambda.1se), coef(lm(Salary~.,data=Hitters))),4)
plot(fit.ridge,xvar="lambda",label=TRUE)
abline(v=log(cv.ridge$lambda.min), col="red", lty=2)
abline(v=log(cv.ridge$lambda.1se), col="red", lty=2)

# To Let us split the data in training and validation set
n <- nrow(data)
training_split <- round(2*n/3, 0)

set.seed(26)
train <- sample(n, training_split)
cv.ridge <- cv.glmnet(X[train,], y[train], alpha = 0, nfolds = 10)

plot(cv.ridge) # the tradeoff is clearly visible as the MSE decreases and then grows again
cv.ridge$lambda.min
log(cv.ridge$lambda.min)
cv.ridge$lambda.1se
log(cv.ridge$lambda.1se)

ridgepred.min <- predict(cv.ridge, s = cv.ridge$lambda.min, newx = X[-train,])
ridgepred.1se <- predict(cv.ridge, s = cv.ridge$lambda.1se, newx = X[-train,])

mean((ridgepred.min-y[-train])^2)
mean((ridgepred.1se-y[-train])^2)







#Now we fit the models using the lasso estimator
#in glmnet default `alpha=1`


fit.lasso <- glmnet(x,y, alpha = 1)
plot(fit.lasso, xvar = "lambda", label=TRUE)
plot(fit.lasso, xvar = "lambda", label=TRUE, ylim = c(-5,5))
names(fit.lasso)
fit.lasso$lambda

# matrix of coefficients
dim(fit.lasso$beta)
fit.lasso$lambda[28]
fit.lasso$beta[,28]

# cross-validation on the train set
cv.lasso <- cv.glmnet(x[train,], y[train], nfolds = 10)
plot(cv.lasso)
coef(cv.lasso) # corresp to lambda.1se
coef(cv.lasso, s=cv.lasso$lambda.min) # corresp to lambda.min

cv.lasso$nzero # non-zero coeff
plot(cv.lasso$lambda, cv.lasso$nzero, pch=20, col="red")

lassopred.min <- predict(cv.lasso,s=cv.lasso$lambda.min,newx=x[-train,])
mean((lassopred.min-y[-train])^2)
mean((ridgepred.min-y[-train])^2)
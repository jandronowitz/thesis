# Julia Andronowitz
# Classification Methods Using Support Vector Machines
# Support Vector Machine


# loading svm library (run first line if not installed)
# install.packages("e1071")
library(e1071)

# defining the hyperplane function
hyperplane <- function(P,data,x,z=0) {
  alphas <- -1*P$coefs
  svs <- data[P$index,]
  c <- P$rho - z
  a <- sum(t(alphas)*data[P$index,][1])
  b <- sum(t(alphas)*data[P$index,][2])
  (-c-a*x)/b
  cat(a,b,c)
}

# setting initial number of observations in each set
n <- 200

# calculating x-variable using three randomly generated sets with overlap
set.seed(1000)

svmdatax1a <- matrix(rnorm(n, 0.25,0.3), ncol=1)
svmdatax1b <- matrix(rnorm(n,0.75,0.3), ncol=1)
svmdatax1 <- cbind(svmdatax1a,svmdatax1b)

svmdatax2 <- matrix(rnorm(n*2,-0.5,0.4), ncol=2)

svmdatax3a <- matrix(rnorm(n,1.5,0.3), ncol=1)
svmdatax3b <- matrix(rnorm(n,1,0.3), ncol=1)
svmdatax3 <- cbind(svmdatax3a,svmdatax3b)

svmdatax <- rbind(svmdatax1,svmdatax2,svmdatax3)

# assigning labels to both classes
svmdatay <- c(rep(1,n), rep(2,n), rep(3,n))

# plotting the data
plot(svmdatax, col=(11-svmdatay),xlab="",ylab="",pch=16)

# creating data frame 
svmdata <- data.frame (x = svmdatax, y = factor(svmdatay))

# using different costs
svmfit.00001 <- svm (y~., data = svmdata , kernel="radial", cost = 0.00001, scale = FALSE)
svmfit.001 <- svm (y~., data = svmdata , kernel="radial", cost = 0.001, scale = FALSE)
svmfit1 <- svm (y~., data = svmdata , kernel="radial", cost = 1, scale = FALSE)
svmfit1000 <- svm (y~., data = svmdata , kernel="radial", cost = 1000, scale = FALSE)
svmfit100000 <- svm (y~., data = svmdata , kernel="radial", cost = 100000, scale = FALSE)
svmfit10000000 <- svm (y~., data = svmdata , kernel="radial", cost = 10000000, scale = FALSE)

# plots of different costs
plot(svmfit.00001,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit.001,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit1,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit1000,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit100000,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit10000000,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))

# using different gammas
svmfit.01g <- svm (y~., data = svmdata , kernel="radial", cost = 1, gamma = 0.01, scale = FALSE)
svmfit.1g <- svm (y~., data = svmdata , kernel="radial", cost = 1, gamma = 0.1, scale = FALSE)
svmfit1g <- svm (y~., data = svmdata , kernel="radial", cost = 1, gamma = 1, scale = FALSE)
svmfit10g <- svm (y~., data = svmdata , kernel="radial", cost = 1, gamma = 10, scale = FALSE)
svmfit100g <- svm (y~., data = svmdata , kernel="radial", cost = 1, gamma = 100, scale = FALSE)
svmfit1000g <- svm (y~., data = svmdata , kernel="radial", cost = 1, gamma = 1000, scale = FALSE)

# plots of different gammas
plot(svmfit.01g,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit.1g,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit1g,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit10g,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit100g,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))
plot(svmfit1000g,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))

# using tune function to find the best model
tune <- tune(svm,y~.,data=svmdata, 
             ranges = list(gamma = 2^(-2:10), cost = 2^(-2:10)), tunecontrol = tune.control(sampling = "fix"))

tune$best.parameters
tune$best.performance
 # tune$performances
    # run to see individual performances

# computing svm function with best gamma and cost
svmfit <- svm (y~., data = svmdata , kernel="radial", cost = 0.25, gamma=0.5, scale = FALSE)

# looking at statistics
summary(svmfit)
svmfit$index
svmfit$rho
svmfit$coefs

# plotting svm function graph
plot(svmfit,svmdata,color.palette = terrain.colors,symbolPalette = c("red","black","darkgray"))

# looking at how a,b,c values differ 
hyperplane(svmfit,svmdata,svmdatax)
hyperplane(svmfit1,svmdata,svmdatax)
hyperplane(svmfit1000000,svmdata,svmdatax)


help(tune)
help(svm)


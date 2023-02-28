# Julia Andronowitz
# Classification Methods Using Support Vector Machines
# Support Vector Classifier


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

# calculating x-variable using two randomly generated sets with overlap
set.seed(100)
svcdatax1 <- matrix(rnorm(n*2, 0.5,0.3), ncol=2)
svcdatax2 <- matrix(rnorm(n*2,-0.5,0.3), ncol=2)
svcdatax <- rbind(svcdatax1,svcdatax2)

# assigning labels to both classes
svcdatay <- c(rep(-1,n), rep(1,n))

# plotting the data
plot(svcdatax, col=(3-svcdatay),xlab="",ylab="",pch=16)

# adding line to visualize a hyperplane
abline(0,-1,col="gray",lty="dashed")

# creating data frame 
svcdata <- data.frame (x = svcdatax, y = factor(svcdatay))

# computing svm function
svmfit <- svm (y~., data = svcdata , kernel="linear", cost = 1000, scale = FALSE)

# looking at statistics
summary(svmfit)
svmfit$index
svmfit$rho
svmfit$coefs

# plotting svm function graph
plot(svmfit,svcdata)

# calculating the hyperplane and supporting hyperplanes
plt0 <- hyperplane(svmfit,svcdata,svcdatax,0)
plt1 <- hyperplane(svmfit,svcdata,svcdatax,1)
plt2 <- hyperplane(svmfit,svcdata,svcdatax,-1)

#FIX*****
# plotting the data with hyperplanes
plot(svcdatax,col=(3-svcdatay),xlab="",ylab="",pch=16)
lines(svcdatax,plt0,col='black')
lines(svcdatax,plt1,col='gray')
lines(svcdatax,plt2,col='gray')

# using cost of 1
svmfit1 <- svm (y~., data = svcdata , kernel="linear", cost = 0.1, scale = FALSE)
summary(svmfit1)
  # results in 36 support vectors
plot(svmfit1,svcdata)


# using cost of 100
svmfit100 <- svm (y~., data = svcdata , kernel="linear", cost = 100, scale = FALSE)
summary(svmfit100)
  # results in 21 support vectors

# using cost of 1000
svmfit1000 <- svm (y~., data = svcdata , kernel="linear", cost = 1000, scale = FALSE)
summary(svmfit1000)
  # results in 20 support vectors

# using cost of 10000
svmfit10000 <- svm (y~., data = svcdata , kernel="linear", cost = 10000, scale = FALSE)
summary(svmfit10000)
  # results in 20 support vectors

#calculating sum of slack variables
sum(1 - svcdata[svmfit$index,]$x.1*svmfit$coefs + svmfit$rho)
sum(1 - svcdata[svmfit100$index,]$x.1*svmfit100$coefs + svmfit100$rho)
sum(1 - svcdata[svmfit1000$index,]$x.1*svmfit1000$coefs + svmfit1000$rho)
sum(1 - svcdata[svmfit10000$index,]$x.1*svmfit10000$coefs + svmfit10000$rho)

# get a,b,c values
hyperplane(svmfit1,svcdata,svcdatax)
hyperplane(svmfit100,svcdata,svcdatax)
hyperplane(svmfit1000,svcdata,svcdatax)
hyperplane(svmfit10000,svcdata,svcdatax)

# tune the function to find the best model with various costs
tune <- tune(svm,y~.,data=svcdata, 
             ranges = list(cost = 2^(-2:10)), tunecontrol = tune.control(sampling = "fix"))

tune$best.parameters
  # we see a cost of 0.25 gives the best model
tune$best.performance
  # this is the model error

# final svc model
svmfit <- svm (y~., data = svcdata , kernel="linear", cost = 0.25, scale = FALSE)
plot(svmfit,svcdata)

help(svm)

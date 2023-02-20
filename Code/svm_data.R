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
}

# setting initial number of observations in each set
n <- 200

# calculating x-variable using two randomly generated sets with overlap
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
plot(svmdatax, col=(3-svmdatay),xlab="",ylab="",pch=16)

# creating data frame 
svmdata <- data.frame (x = svmdatax, y = factor(svmdatay))

# computing svm function
svmfit <- svm (y~., data = svmdata , kernel="radial", cost = 1000, scale = FALSE)

# looking at statistics
summary(svmfit)
svmfit$index
svmfit$rho
svmfit$coefs

# plotting svm function graph
plot(svmfit,svmdata,color.palette = terrain.colors,symbolPalette = c("yellow","red","black"))


help(svm)

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
}

# setting initial number of observations in each set
n <- 200

# calculating x-variable using two randomly generated sets with overlap
set.seed(100)

mmcdatax1 <- matrix(rnorm(n*2, 1,0.3), ncol=2)
mmcdatax2 <- matrix(rnorm(n*2,-1,0.3), ncol=2)
mmcdatax <- rbind(mmcdatax1,mmcdatax2)

# assigning labels to both classes
mmcdatay <- c(rep(-1,n), rep(1,n))

# plotting the data
plot(mmcdatax, col=(3-mmcdatay),xlab="",ylab="",pch=16)

# adding line to visualize a hyperplane
abline(0,-1,col="gray",lty="dashed")

# creating data frame 
mmcdata <- data.frame (x = mmcdatax, y = factor(mmcdatay))

# computing svm function
svmfit <- svm (y~., data = mmcdata , kernel="linear", cost = 1000, scale = FALSE)

# looking at statistics
summary(svmfit)
svmfit$index
svmfit$rho
svmfit$coefs

# plotting svm function graph
plot(svmfit,mmcdata)

# calculating the hyperplane and supporting hyperplanes
plt0 <- hyperplane(svmfit,mmcdata,mmcdatax,0)
plt1 <- hyperplane(svmfit,mmcdata,mmcdatax,1)
plt2 <- hyperplane(svmfit,mmcdata,mmcdatax,-1)

# plotting the data with hyperplanes
plot(mmcdatax,col=(3-mmcdatay),xlab="",ylab="",pch=16)
lines(mmcdatax,plt0,col='black')
lines(mmcdatax,plt1,col='gray')
lines(mmcdatax,plt2,col='gray')

help(svm)

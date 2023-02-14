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

# setting initial number of observations
n <- 200

# calculating x-variable using two randomly generated sets with overlap
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
svcdata <- data.frame (x = svcdatax, y = as.factor (svcdatay))

# computing svm function
svmfit <- svm (svcdatay~., data = svcdata , kernel="linear", cost = 1000, scale = FALSE)

# plotting svm function
plot(svmfit,svcdata)

hyperplane(svmfit,dat,x,0)

plt0 <- hyperplane(svmfit,dat,x,0)
plt1 <- hyperplane(svmfit,dat,x,1)
plt2 <- hyperplane(svmfit,dat,x,-1)

plot(newx,col=(3-y),xlab="",ylab="",sub="Figure 2.1")
lines(x,plt0,col='black')
lines(x,plt1,col='gray')
lines(x,plt2,col='gray')

summary(svmfit)
svmfit$index
svmfit$rho
svmfit$coefs

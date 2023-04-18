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
abline(h=0,v=0,col="gray",lty="dotted")
abline(a=0,b=-1,col="black",lty="dashed")

# creating data frame 
mmcdata <- data.frame (x = mmcdatax, y = factor(mmcdatay))

# splitting into training and test data
library(caTools)
set.seed(100)

# using 80% of the data in the training set and the other 20% in the test set
sample <- sample.split(mmcdata$x.1, SplitRatio = 0.8)
train  <- subset(mmcdata, sample == TRUE)
test   <- subset(mmcdata, sample == FALSE)

# computing svm function
svmfit <- svm (y~., data = train , kernel="linear", cost = 1000, scale = FALSE)

# looking at statistics
summary(svmfit)
svmfit$index
svmfit$rho
svmfit$coefs

# plotting svm function graph
plot(svmfit,train)

# predicting the test set
y_pred <- predict(svmfit, newdata = test[-3])

# viewing the confusion matrix; no misclassified points
cm = table(test[,3], y_pred)
cm

# set x and y variables into separate dataframes
trainx <- cbind(train[,1],train[,2])
trainy <- as.numeric(unlist(train[3]))

# convert the y-values back to -1 and 1
trainy[trainy == 1] <- -1
trainy[trainy == 2] <- 1

# calculating the hyperplane and supporting hyperplanes
plt0 <- hyperplane(svmfit,train,trainx,0)
plt1 <- hyperplane(svmfit,train,trainx,1)
plt2 <- hyperplane(svmfit,train,trainx,-1)

# plotting the data with hyperplanes
plot(trainx,col=(3-trainy),xlab="",ylab="",pch=16)
lines(trainx,plt0,col='black')
lines(trainx,plt1,col='gray',lty='dashed')
lines(trainx,plt2,col='gray',lty='dashed')



help(svm)

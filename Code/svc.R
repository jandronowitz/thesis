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

# defining hyperplane function that returns a,b,c values
hyperplane_vals <- function(P,data,x,z=0) {
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

# splitting into training and test data
library(caTools)
set.seed(100)

# using 80% of the data in the training set and the other 20% in the test set
sample <- sample.split(svcdata$x.1, SplitRatio = 0.8)
train  <- subset(svcdata, sample == TRUE)
test   <- subset(svcdata, sample == FALSE)

# computing svm function
svmfit <- svm (y~., data = train , kernel="linear", cost = 1000, scale = FALSE)

# looking at statistics
summary(svmfit)
svmfit$index
svmfit$rho
svmfit$coefs

# plotting svm function graph
plot(svmfit,train)

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
lines(trainx,plt1,col='gray')
lines(trainx,plt2,col='gray')

# predicting the test set
y_pred <- predict(svmfit, newdata = test[-3])

# viewing the confusion matrix; no misclassified points
cm = table(test[,3], y_pred)
cm

# using cost of 0.01
svmfit01 <- svm (y~., data = train , kernel="linear", cost = .01, scale = FALSE)
summary(svmfit01)
  # results in 224 support vectors

# using cost of 1
svmfit1 <- svm (y~., data = train , kernel="linear", cost = 1, scale = FALSE)
summary(svmfit1)
  # results in 35 support vectors

# using cost of 100
svmfit100 <- svm (y~., data = train , kernel="linear", cost = 100, scale = FALSE)
summary(svmfit100)
  # results in 21 support vectors

# using cost of 1000
svmfit1000 <- svm (y~., data = train , kernel="linear", cost = 1000, scale = FALSE)
summary(svmfit1000)
  # results in 21 support vectors

# exploration; calculating sum of slack variables
sum(1 - train[svmfit$index,]$x.1*svmfit$coefs + svmfit$rho)
sum(1 - train[svmfit100$index,]$x.1*svmfit100$coefs + svmfit100$rho)
sum(1 - train[svmfit1000$index,]$x.1*svmfit1000$coefs + svmfit1000$rho)
sum(1 - train[svmfit10000$index,]$x.1*svmfit10000$coefs + svmfit10000$rho)

# exploration; get a,b,c values
hyperplane_vals(svmfit1,train,trainx)
hyperplane_vals(svmfit100,train,trainx)
hyperplane_vals(svmfit1000,train,trainx)
hyperplane_vals(svmfit10000,train,trainx)

# tune the function to find the best model with various costs
tune <- tune(svm, y~., data = train, 
             ranges = list(cost = 2^(-2:10)), tunecontrol = tune.control(sampling = "fix"))

# we see a cost of 0.25 gives the best model
tune$best.parameters

# calculating the model error
tune$best.performance

# final svc model
svmfit <- svm (y~., data = train , kernel="linear", cost = 0.25, scale = FALSE)
plot(svmfit,train)
summary((svmfit))

# predicting the test set
y_pred <- predict(svmfit, newdata = test[-3])

# viewing the confusion matrix; no misclassified points
cm = table(test[,3], y_pred)
cm

help(tune)
help(svm)

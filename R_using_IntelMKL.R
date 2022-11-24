## https://jooskorstanje.com/speed-benchmark-r.html


# Without the Intel OneApi this took 6:30.
# With the Intel OneApi installed..

base::remove(list = ls())

source("C:/Users/droll/Documents/R/startup/eeros_functions.R")

using_packages("caret", "nnet", "MASS", "e1071" )  # dropped "doParallel"


## The Data ------------

header <- c("sepal_length", "sepal_width", "petal_length", "petal_width", "class")
iris <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
names(iris) <- header

head(iris)


sample <- sample.int(n = nrow(iris), size = floor(.75*nrow(iris)), replace = F)
train <- iris[sample, ]
test  <- iris[-sample, ]

## Fitting the models ---- 
## 
## 
# cl <- makePSOCKcluster(3) # Using pure Intel MKL instead
# registerDoParallel(cl)    # Using pure Intel MKL instead

### Fitting a logistic regression ------------
### using the nnet interface, because contrary to mlogit it doesnt require a data reshape
lr <- multinom(class ~ ., data = train)

### Fitting the LDA model ------------
lda_model <- lda(class ~ ., data = train)

### Fitting the knn ------- 
### with crossval hyperparameter tuning
man_grid <- expand.grid(k = c(1:10))
ctrl <- trainControl(method="cv", number = 5)
knn <- train(class ~ ., data = train, method = "knn", trControl = ctrl, tuneGrid = man_grid)

### Fitting the SVM ------------
man_grid <- expand.grid(cost = c(1:10))
ctrl <- trainControl(method="cv", number = 5)
svm <- train(class ~ ., data = train, method = "svmLinear2", trControl = ctrl, tuneGrid = man_grid)

# stopCluster(cl)  # Using pure Intel MKL instead


## Evaluate ----------------
## 
lr_accuracy <- sum(predict(lr, test) == test$class) / nrow(test)
lda_accuracy <- sum(predict(lda_model, test)$class == test$class) / nrow(test)
knn_accuracy <- sum(predict(knn, test) == test$class) / nrow(test)
svm_accuracy <- sum(predict(svm, test) == test$class) / nrow(test)
print(paste(lr_accuracy, ",", lda_accuracy, ",", knn_accuracy, ",",  svm_accuracy)) # 


## The Speed Benchmark ------

main <- function() {
  header <- c("sepal_length", "sepal_width", "petal_length", "petal_width", "class")
  iris <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")
  names(iris) <- header
  sample <- sample.int(n = nrow(iris), size = floor(.75*nrow(iris)), replace = F)
  train <- iris[sample, ]
  test  <- iris[-sample, ]
  
  # cl <- makePSOCKcluster(3) # Using pure Intel MKL instead
  # registerDoParallel(cl)    # Using pure Intel MKL instead
  
  lr <- multinom(class ~ ., data = train)
  lda_model <- lda(class ~ ., data = train)
  
  man_grid <- expand.grid(k = c(1:10))
  ctrl <- trainControl(method="cv", number = 5)
  knn <- train(class ~ ., data = train, method = "knn", trControl = ctrl, tuneGrid = man_grid)
  
  man_grid <- expand.grid(cost = c(1:10))
  ctrl <- trainControl(method="cv", number = 5)
  svm <- train(class ~ ., data = train, method = "svmLinear2", trControl = ctrl, tuneGrid = man_grid)
  
  # stopCluster(cl)           # Using pure Intel MKL instead
  
  # Evaluate
  lr_accuracy <- sum(predict(lr, test) == test$class) / nrow(test)
  lda_accuracy <- sum(predict(lda_model, test)$class == test$class) / nrow(test)
  knn_accuracy <- sum(predict(knn, test) == test$class) / nrow(test)
  svm_accuracy <- sum(predict(svm, test) == test$class) / nrow(test)
  print(paste(lr_accuracy, ",", lda_accuracy, ",", knn_accuracy, ",", svm_accuracy))
}


now <- Sys.time()
for (i in c(1:100)) {
  print(i)
  main()
}
now2 <- Sys.time()
print(now2 - now)


paste('Time per loop in seconds is ', ((now2 - now) / 100) * 60)



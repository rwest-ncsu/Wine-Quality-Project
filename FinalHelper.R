#Load required libraries
library(ggplot2)
library(e1071)
library(randomForest)
library(MASS)
library(caret)
library(dplyr)
library(class)
library(FNN)
library(tree)
library(gbm)
library(knitr)
library(tidyr)

#Helper functions
#KNN classification
make_knn_pred = function(k = 1, train_X, test_X, train_Y, test_Y) {
  pred = knn(train_X, test_X, train_Y, k = k)
  mean(test_Y!=pred)}
#KNN Regression
rmse = function(actual, predicted) {
  sqrt(mean((actual - predicted) ^ 2))
}
make_knn_pred_Reg = function(k = 1, train_X, test_X, train_Y, test_Y) {
  pred = knn.reg(train = train_X, 
                 test = test_X, 
                 y = train_Y, k = k)$pred
  act = test_Y
  rmse(predicted = pred, actual = act)
}
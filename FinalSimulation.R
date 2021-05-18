#Final Project simulation

#Load helper functions
source("FinalHelper.r")

#Read in data
wine = read.csv("winequality-red.csv", sep=";")

#Split data groups based on Quality
split = factor((wine$quality > 5), labels = c("Bad", "Good"))

wineTotal = wine %>% 
  mutate(split=split)


#Start simulation
nSim = 10
testError = matrix(0, nrow=nSim, ncol=9)
colnames(testError) = c("Logistic",
                      "Tree", 
                      "Bagged", 
                      "RF",
                      "GBM",
                      "LDA",
                      "QDA",
                      "KNN",
                      "SVM")

for(i in 1:nSim){
  #Split data into train and test
  train = sample(1:nrow(wineTotal), size=nrow(wineTotal)*0.8, replace = F)
  wineTrain = wineTotal[train, ]%>%dplyr::select(-quality)
  wineTest = wineTotal[-train, ]%>%dplyr::select(-quality)
  
  
  #LOGISTIC REGRESSION
  outliers = which(wineTotal$volatile.acidity > 1.2)
  outliers = append(outliers, which(wineTotal$chlorides > 0.25))
  outliers = append(outliers, which(wineTotal$sulphates > 1.5))
  logistic = wineTotal[-outliers,]
  # Create a new variable and remove variables
  logistic = logistic %>%
    mutate(ratio.sulfur.dioxide = free.sulfur.dioxide/total.sulfur.dioxide) %>%
    dplyr::select(-c(fixed.acidity,free.sulfur.dioxide,total.sulfur.dioxide,citric.acid,density,residual.sugar))
  train.logistic = logistic[train,]%>%na.omit()
  test.logistic = logistic[-train,]%>%na.omit()
  #Create the logistic regression model
  glm.fit.wine = glm(split ~ volatile.acidity+pH+sulphates+I(sulphates^2)+
                       I(sulphates^3)+alcohol+ratio.sulfur.dioxide,
                     data=train.logistic, family = binomial)
  #Logistic Prediction
  glm.probs = predict(glm.fit.wine,type='response', newdata = test.logistic)
  glm.pred = factor((glm.probs > 0.5), labels = c("Bad", "Good"))
  #Logistic MSE
  testError[i, 1] = mean(glm.pred != test.logistic$split)
  
  
  #SINGLE TREE
  #Logistic regression and tree methods removed the same observations
  trees = wineTotal[-outliers, ]%>%dplyr::select(-quality)
  treeTrain = trees[train, ]%>%na.omit()
  treeTest = trees[-train, ]%>%na.omit()
  #Build and predict on single tree
  treeMod= tree(split~., data=treeTrain)
  treePred = predict(treeMod, newdata=treeTest, type = "class")
  #Single tree MSE
  testError[i, 2] = mean(treePred != treeTest$split)
  
  
  #BAGGED TREE
  bagMod = randomForest(split ~ ., data=treeTrain, mtry=11, importance=T)
  bagPred = predict(bagMod, newdata=treeTest, type="class")
  #Bagged tree MSE
  testError[i, 3] = mean(bagPred != treeTest$split)
  
  
  #RANDOM FOREST
  control=trainControl(method="cv", number=5, search="grid")
  tunegrid=expand.grid(mtry=c(1:11))
  #Build RF Model by Cross Validation
  rf_gridsearch=train(split~., data=treeTrain, method="rf", metric="Accuracy",
                      tuneGrid=tunegrid, trControl=control)
  
  rfMod=rf_gridsearch$finalModel
  rfPred = predict(rfMod, newdata=treeTest)
  #RF MSE
  testError[i, 4] = mean(rfPred != treeTest$split)
  
  
  #GBM 
  control=trainControl(method="cv", number=5, search="grid")
  tunegrid=expand.grid(n.trees=1000,
                       interaction.depth=5,
                       shrinkage=c(0.001,0.005,0.01,0.015,0.02, 0.03, 0.05, 0.1),
                       n.minobsinnode=3)
  #Fit GBM by Cross validation
  gb_gridsearch=train(split~., data=treeTrain, method="gbm", metric="Accuracy",
                      tuneGrid=tunegrid, trControl=control, verbose=F)
  #Grab best model
  boost.data=gbm(split~., data=treeTrain, distribution="multinomial", n.trees=10000,
                 shrinkage=gb_gridsearch$bestTune$shrinkage, interaction.depth=4)
  #GBM Prediction 
  gbmPred = predict.gbm(object = boost.data,
                        newdata = treeTest,
                        n.trees = 500,
                        type = "response")
  labels = colnames(gbmPred)[apply(gbmPred, 1, which.max)]
  #GBM MSE
  testError[i, 5] = mean(labels != treeTest$split)
  
  
  #LDA
  #Transformations in an attempt to meet Normality assumptions
  LDA = wineTotal
  LDA$split = wineTotal$split
  LDA$volatile.acidity = wineTotal$volatile.acidity
  LDA$chlorides = log10(wineTotal$chlorides)
  LDA$pH = wineTotal$pH
  LDA$sulphates = log10(wineTotal$sulphates)
  LDA$alcohol = sqrt(wineTotal$alcohol)
  LDA$ratio_sulfur.dioxide = wineTotal$free.sulfur.dioxide /
    wineTotal$total.sulfur.dioxide
  LDA = subset(LDA, select = c(volatile.acidity, chlorides,
                               pH, sulphates, alcohol, 
                               ratio_sulfur.dioxide, split))
  LDATrain = LDA[train, ]
  LDATest = LDA[-train, ]
  #LDA Fit
  lda.fit = lda(split~. , data = LDATrain)
  lda.pred = predict(lda.fit, LDATest)$class
  #LDA MSE
  testError[i, 6] = mean(lda.pred != LDATest$split)
  #QDA Fit
  qda.fit = qda(split~. , data = LDATrain)
  qda.pred <- predict(qda.fit, LDATest)$class
  #QDA MSE
  testError[i, 7] = mean(qda.pred != LDATest$split)
  
  
  #KNN
  #Variables to include after selection performed
  preds = c(2,5,6,7,9,10,11)
  X_trn = wineTrain[ ,preds]
  X_tst  = wineTest[ ,preds]
  #Grid over which to search for optimal K 
  k = c(1, 3, 5, 10, 25, 50, 100)
  knn_tst_rmse = sapply(k, make_knn_pred, 
                        train_X = X_trn, 
                        test_X = X_tst,
                        train_Y = wineTrain$split, 
                        test_Y = wineTest$split)
  # determine "best" k
  best_k = k[which.min(knn_tst_rmse)]
  #KNN MSE 
  testError[i, 8] = make_knn_pred(k=best_k,
                      train_X = X_trn, 
                      test_X = X_tst, 
                      train_Y = wineTrain$split, 
                      test_Y = wineTest$split)
  
  
  #SVM
  radialSVM = tune(svm, split ~., data=wineTrain, kernel="radial", scale=T, 
                   ranges=list(
                     cost=c(0.001, 0.01, 0.1, 1), 
                     gamma=c(0.5, 1, 2, 3, 4)
                   ))
  radialSVMPred = predict(radialSVM$best.model, newdata=wineTest)
  #SVM MSE
  testError[i, 9] = mean(radialSVMPred != wineTest$split)
}

#Create the Image of MSE
errorPlot = as.data.frame(testError) %>%
  gather(key="Method", value = "MSE") %>%
  mutate(sim = c(rep(1:nSim, 9)))

ggplot(data=errorPlot, aes(x=sim, y=MSE))+
  geom_point(aes(color=Method))+
  geom_line(aes(color=Method))+
  labs(title="Repeated Simulation of Classification Methods")+
  ylab("Test Misclassification Rate")+
  xlab("Simulation Number")+
  scale_x_continuous(breaks = 1:nSim)

kable(sort(colMeans(testError))) #Averages

kable(testError) #Observations














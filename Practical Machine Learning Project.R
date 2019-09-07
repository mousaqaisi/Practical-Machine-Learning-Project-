library(caret)
library(ggplot2)
library(rattle)
library(randomForest)
TrainData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"),header=TRUE)
dim(TrainData)
TestData <- read.csv(url("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"),header=TRUE)
dim(TestData)
str(TrainData)
indColToRemove <- which(colSums(is.na(TrainData) |TrainData=="")>0.9*dim(TrainData)[1]) 
TrainDataClean <- TrainData[,-indColToRemove]
TrainDataClean <- TrainDataClean[,-c(1:7)]
dim(TrainDataClean)
str(TrainDataClean)
indColToRemove <- which(colSums(is.na(TestData) |TestData=="")>0.9*dim(TestData)[1]) 
TestDataClean <- TestData[,-indColToRemove]
TestDataClean <- TestDataClean[,-1]
dim(TestDataClean)
str(TestDataClean)

set.seed(12345)
inTrain1 <- createDataPartition(TrainDataClean$classe, p=0.75, list=FALSE)
Train1 <- TrainDataClean[inTrain1,]
Test1 <- TrainDataClean[-inTrain1,]
dim(Train1)

#Train with classification tree
trControl <- trainControl(method="cv", number=5)
model_CT <- train(classe~., data=Train1, method="rpart", trControl=trControl)
fancyRpartPlot(model_CT$finalModel)
trainpred <- predict(model_CT,newdata=Test1)
confMatCT <- confusionMatrix(Test1$classe,trainpred)
confMatCT$table
confMatCT$overall[1]
# Accuracy
# As we can notice that the accuracy of this first model is very low (about 55%). This means that the outcome class will not be predicted very well by the other predictors.
#Train with random forests
model_RF <- train(classe~., data=Train1, method="rf", trControl=trControl, verbose=FALSE)
## Loading required package: randomForest

## Type rfNews() to see new features/changes/bug fixes.
rfNews()
print(model_RF)
plot(model_RF,main="Accuracy of Random forest model by number of predictors")
trainpred <- predict(model_RF,newdata=Test1)

confMatRF <- confusionMatrix(Test1$classe,trainpred)

# display confusion matrix and model accuracy
confMatRF$table
confMatRF$overall[1]
names(model_RF$finalModel)
model_RF$finalModel$classes
plot(model_RF$finalModel,main="Model error of Random forest model by number of trees")
MostImpVars <- varImp(model_RF)
MostImpVars
#With random forest, we reach an accuracy of 99.3% using cross-validation with 5 steps. This is very good
#Train with gradient boosting method
library(gbm)
model_GBM <- train(classe~., data=Train1, method="gbm", trControl=trControl, verbose=FALSE)
library(survival)
library(splines)
library(parallel)
print(model_GBM)
plot(model_GBM)
trainpred <- predict(model_GBM,newdata=Test1)

confMatGBM <- confusionMatrix(Test1$classe,trainpred)
confMatGBM$table
confMatGBM$overall[1]
#Conclusion
#This shows that the random forest model is the best one. We will then use it to predict the values of classe for the test data set.
FinalTestPred <- predict(model_RF,newdata=TestDataClean)
FinalTestPred



Practical Machine Learning Course Project
========================================================
## Wearable devices. Measuring quality instead of quantity. 

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. 

Source data for the project is recieved from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. The goal of the project is to predict the manner in which they did the exercise using the data from the accelerometers and other available in the dataset.

### Code and comments

We model and predict using functions from the caret package

```r
library(caret)
```

To keep the research reproducible the links to the source data are added to the document.

```r
#Download data file from the web
fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(fileUrl, destfile = "./data/pml-training.csv", method = "curl")

fileUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(fileUrl, destfile = "./data/pml-testing.csv", method = "curl")

#Read data to dataframe
training_dataset <- read.csv("./data/pml-training.csv", stringsAsFactors = FALSE)
testing_dataset <- read.csv("./data/pml-testing.csv", stringsAsFactors = FALSE)
```

Primary data preprocessing: 
we substitude "#DIV/0!" expression by NA, select only numeric columns and make sure all data in them is numeric.


```r
training_dataset[training_dataset=="#DIV/0!"] = NA
training_short <- training_dataset[c(-1,-2,-3,-4,-5,-6,-160)]
training_num <- sapply(training_short, as.numeric)
training <- data.frame(training_dataset[160], training_num)
```

Next step - remove all columns, containg more than 1000 NAs 
(for this dataframe it just removes all columns with NAs)

```r
nacols <- sapply(training, function(x) { length(which(is.na(x))) })
logcols <- nacols < 1000
shtrain <- training [logcols]
```

Adding a numeric variable

```r
shtrain$classe <- as.factor(shtrain$classe)
shtrain$classe_num <- as.numeric(shtrain$classe)
```

Select 20% of the full sample to experiment with the data

```r
inTrain <- createDataPartition(y=shtrain$classe, p=0.2, list=FALSE)
train_final <- shtrain[inTrain,-1]
```

Build a general linear model predicting the numeric variable, corresponding to the factor variable we want to predict.
We do this as while have too many predictors we want to run a "cheap"" algorithm. 

```r
modelFit <- train(classe_num ~., data=train_final, method="glm")
```

The model built helps us to identify the most important variables

```r
varImp(modelFit)
```

```
## glm variable importance
## 
##   only 20 most important variables shown (out of 53)
## 
##                      Overall
## magnet_dumbbell_z      100.0
## accel_forearm_z         67.7
## magnet_dumbbell_x       50.6
## accel_arm_z             37.7
## yaw_dumbbell            35.4
## magnet_belt_y           34.9
## pitch_forearm           32.7
## accel_belt_y            29.5
## accel_dumbbell_x        27.6
## pitch_belt              26.9
## total_accel_forearm     26.7
## total_accel_dumbbell    25.4
## magnet_dumbbell_y       23.4
## roll_dumbbell           18.5
## magnet_forearm_x        17.3
## roll_belt               17.1
## gyros_arm_x             17.1
## total_accel_arm         17.1
## pitch_arm               16.3
## gyros_dumbbell_y        16.3
```

Create a new dataframe, that contains only main variables.

```r
train_fin_20 <- shtrain[inTrain,c("classe","magnet_dumbbell_z","accel_forearm_z","magnet_dumbbell_x","yaw_dumbbell","accel_arm_z","pitch_forearm","accel_belt_y","magnet_belt_y","accel_dumbbell_x","total_accel_forearm","pitch_belt","roll_dumbbell","pitch_arm","roll_belt","magnet_forearm_x","total_accel_arm","magnet_forearm_y","yaw_belt","gyros_dumbbell_y")]
```
    
Apply random forest algorithm

```r
modelFit <- train(classe ~ ., data=train_fin_20, method="rf")
```

### Cross validation

Prepare the testing set and apply the model to it

```r
testing <- shtrain[-inTrain,]
predictions <- predict(modelFit,newdata=testing)
```

Show confusion matrix

```r
confusionMatrix(testing$classe, predict(modelFit,newdata=testing))
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.1.1
```

```
## randomForest 4.6-10
## Type rfNews() to see new features/changes/bug fixes.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4438    5    7   11    3
##          B  108 2862   57   10    0
##          C    7   75 2630   24    1
##          D   12    2   74 2481    3
##          E    1    9    4   15 2856
## 
## Overall Statistics
##                                        
##                Accuracy : 0.973        
##                  95% CI : (0.97, 0.975)
##     No Information Rate : 0.291        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.965        
##  Mcnemar's Test P-Value : <2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.972    0.969    0.949    0.976    0.998
## Specificity             0.998    0.986    0.992    0.993    0.998
## Pos Pred Value          0.994    0.942    0.961    0.965    0.990
## Neg Pred Value          0.989    0.993    0.989    0.995    0.999
## Prevalence              0.291    0.188    0.177    0.162    0.182
## Detection Rate          0.283    0.182    0.168    0.158    0.182
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.985    0.978    0.970    0.985    0.998
```

#### Accuracy : 0.9634
#### Out of sample error: 0.0366

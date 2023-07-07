library(dplyr)
library(randomForest)
library(caret)
library(tidyverse)
library(recipes)
library(boot)
library(tidymodels)
library(ranger)
library(xgboost)
library(e1071)       #for calculating variable importance
library(rpart)       #for fitting decision trees
library(ipred)
library(SimDesign)

songs <- read.csv('processed_songs.csv', stringsAsFactors = TRUE)
head(songs, 10)

set.seed(11111)
feats <- names(songs)[c(5:11,13:15,17)]
train_songs <- songs %>%
  mutate_if(is.numeric, scale)

training_songs <- sample(1:nrow(train_songs), nrow(train_songs)*.75, replace = FALSE)
train_set <- train_songs[training_songs, c('music_genre', feats)] 
test_set <- train_songs[-training_songs, c('music_genre', feats)] 

songs_rf <- randomForest(music_genre~., data = train_set, mtry = 4)

pred_train <- predict(songs_rf)
pred_test <- predict(songs_rf, test_set)

confusionMatrix(pred_train, as.factor(train_set$music_genre))

confusionMatrix(pred_test, as.factor(test_set$music_genre))

var(as.numeric(pred_test), as.numeric(test_set$music_genre))

bias(as.numeric(pred_test), as.numeric(test_set$music_genre))

train_resp <- train_songs[training_songs, 'music_genre']
test_resp <- train_songs[-training_songs, 'music_genre']

matrix_train_gb <- xgb.DMatrix(data = as.matrix(train_set[,-1]), label = as.integer(as.factor(train_set[,1])))
matrix_test_gb <- xgb.DMatrix(data = as.matrix(test_set[,-1]), label = as.integer(as.factor(test_set[,1])))

model_gb <- xgboost(data = matrix_train_gb, 
                    nrounds = 50,
                    verbose = FALSE,
                    params = list(objective = "multi:softmax",
                                  num_class = 10 + 1))

predict_gb_one <- predict(model_gb, matrix_test_gb)
predict_gb <- levels(as.factor(test_set$music_genre))[predict_gb_one]


confusionMatrix(as.factor(predict_gb), as.factor(test_set$music_genre))

var(as.numeric(predict_gb_one), as.numeric(test_set$music_genre))

bias(as.numeric(predict_gb_one), as.numeric(test_set$music_genre))

gbag <- bagging(music_genre ~ ., data = train_set, coob=TRUE)
predict_bag <- predict(gbag, newdata=test_set)

confusionMatrix(as.factor(predict_bag), as.factor(test_set$music_genre))

var(as.numeric(predict_bag), as.numeric(test_set$music_genre))

bias(as.numeric(predict_bag), as.numeric(test_set$music_genre))

library(tidyverse)
library(caret)
library(nnet)
library(pROC)
library(rpart)
library(rpart.plot)
library(caTools)
library(e1071)
library(magrittr)
library(class)
library(SimDesign)
library(RColorBrewer) 
library(dlookr) 
library(ggcorrplot) 
library(plyr) 
library(dplyr) 
library(cowplot) 
library(rsample) 
library(ggplot2) 
library(party) 
library(pROC) 
library(ROCR) 

multiclass_roc_plot <- function(df, probs) {     
     class.0.probs <- probs[,1]
     class.1.probs <- probs[,2]
     class.2.probs <- probs[,3]
     class.3.probs <- probs[,4]
     class.4.probs <- probs[,5]
     class.5.probs <- probs[,6]
     class.6.probs <- probs[,7]
     class.7.probs <- probs[,8]
     class.8.probs <- probs[,9]
     class.9.probs <- probs[,10]
     
     
     actual.0.class <- as.integer(df$music_genre == "Alternative")
     actual.1.class <- as.integer(df$music_genre == "Anime")
     actual.2.class <- as.integer(df$music_genre == "Blues")
     actual.3.class <- as.integer(df$music_genre == "Classical")
     actual.4.class <- as.integer(df$music_genre == "Country")
     actual.5.class <- as.integer(df$music_genre == "Electronic")
     actual.6.class <- as.integer(df$music_genre == "Hip-Hop")
     actual.7.class <- as.integer(df$music_genre == "Jazz")
     actual.8.class <- as.integer(df$music_genre == "Rap")
     actual.9.class <- as.integer(df$music_genre == "Rock")
    
#   "Alternative","Anime","Blues","Classical","Country","Electronic","Hip-Hop","Jazz","Rap","Rock"
    
     plot(x=NA, y=NA, xlim=c(0,1), ylim=c(0,1),
          ylab='True Positive Rate',
          xlab='False Positive Rate',
          bty='n')

     legend(x = "right",                    
            title = "Genre",
            legend = c("Alternative","Anime","Blues","Classical","Country","Electronic","Hip-Hop","Jazz","Rap","Rock"), # Legend
            col = c(1, 2, 3, 4 ,5, 6, 7, 8, 9, 10),
            lwd = 2)

     title("One vs All ROC Curve for 10 Classes")
     
     pred.0 = prediction(class.0.probs, actual.0.class)
     nbperf.0 = performance(pred.0, "tpr", "fpr")
     
     roc.x = unlist(nbperf.0@x.values)
     roc.y = unlist(nbperf.0@y.values)
     lines(roc.y ~ roc.x, col=0+1, lwd=2)
     

     pred.1 = prediction(class.1.probs, actual.1.class)
     nbperf.1 = performance(pred.1, "tpr", "fpr")
     
     roc.x = unlist(nbperf.1@x.values)
     roc.y = unlist(nbperf.1@y.values)
     lines(roc.y ~ roc.x, col=1+1, lwd=2)
     
     pred.2 = prediction(class.2.probs, actual.2.class)
     nbperf.2 = performance(pred.2, "tpr", "fpr")

     roc.x = unlist(nbperf.2@x.values)
     roc.y = unlist(nbperf.2@y.values)
     lines(roc.y ~ roc.x, col=2+1, lwd=2)
     
     pred.3 = prediction(class.3.probs, actual.3.class)
     nbperf.3 = performance(pred.3, "tpr", "fpr")
     
     roc.x = unlist(nbperf.3@x.values)
     roc.y = unlist(nbperf.3@y.values)
     lines(roc.y ~ roc.x, col=3+1, lwd=2)
     
     pred.4 = prediction(class.4.probs, actual.4.class)
     nbperf.4 = performance(pred.4, "tpr", "fpr")
     
     roc.x = unlist(nbperf.4@x.values)
     roc.y = unlist(nbperf.4@y.values)
     lines(roc.y ~ roc.x, col=4+1, lwd=2)
    
     pred.5 = prediction(class.5.probs, actual.5.class)
     nbperf.5 = performance(pred.5, "tpr", "fpr")
     
     roc.x = unlist(nbperf.5@x.values)
     roc.y = unlist(nbperf.5@y.values)
     lines(roc.y ~ roc.x, col=5+1, lwd=2)

     pred.6 = prediction(class.6.probs, actual.6.class)
     nbperf.6 = performance(pred.6, "tpr", "fpr")
     
     roc.x = unlist(nbperf.6@x.values)
     roc.y = unlist(nbperf.6@y.values)
     lines(roc.y ~ roc.x, col=6+1, lwd=2)
    
     pred.7 = prediction(class.7.probs, actual.7.class)
     nbperf.7 = performance(pred.7, "tpr", "fpr")
     
     roc.x = unlist(nbperf.7@x.values)
     roc.y = unlist(nbperf.7@y.values)
     lines(roc.y ~ roc.x, col=7+1, lwd=2)
    
     pred.8 = prediction(class.8.probs, actual.8.class)
     nbperf.8 = performance(pred.8, "tpr", "fpr")
     
     roc.x = unlist(nbperf.8@x.values)
     roc.y = unlist(nbperf.8@y.values)
     lines(roc.y ~ roc.x, col=8+1, lwd=2)
    
     pred.9 = prediction(class.9.probs, actual.9.class)
     nbperf.9 = performance(pred.9, "tpr", "fpr")
     
     roc.x = unlist(nbperf.9@x.values)
     roc.y = unlist(nbperf.9@y.values)
     lines(roc.y ~ roc.x, col=9+1, lwd=2)
     
     lines(x=c(0,1), c(0,1))
 }


songs <- read.csv('processed_songs.csv', stringsAsFactors = TRUE)
head(songs, 10)

set.seed(11111)
feats <- names(songs)[c(5:11,13:15,17)]
train_songs <- songs %>%
  mutate_if(is.numeric, scale)

training_songs <- sample(1:nrow(train_songs), nrow(train_songs)*.75, replace = FALSE)
train_set <- train_songs[training_songs, c('music_genre', feats)] 
test_set <- train_songs[-training_songs, c('music_genre', feats)] 

feats

str(train_set)

str(test_set)

# importance of variables to be considered
import <-rpart(music_genre~.,data=train_set)
import$variable.importance

# multinomial model (only including important attributes from variable importance)
model_lr <- multinom("music_genre~ acousticness + instrumentalness + energy + speechiness + danceability + valence + 
                    tempo + duration_ms", data=train_set, MaxNWts =1000000)


model_lr_train <- predict(object=model_lr, newdata=train_set, type="class")
confusionMatrix(model_lr_train, as.factor(train_set$music_genre))

model_lr_class <- predict(model_lr, test_set, type='class')
confusionMatrix(model_lr_class, as.factor(test_set$music_genre))

model_lr_test <- predict(model_lr, test_set, type='probs')
roc.multi <- multiclass.roc(test_set$music_genre, model_lr_test)
auc(roc.multi)

multiclass_roc_plot(test_set,model_lr_test)

var(as.numeric(model_lr_class), as.numeric(test_set$music_genre))


bias(as.numeric(model_lr_class), as.numeric(test_set$music_genre))

model_svm <- svm(music_genre~., data = train_set, kernel = "radial", probability = TRUE)
model_svm

feats

model_svm_train <- predict(model_svm, train_set[feats])
confusionMatrix(model_svm_train, train_set$music_genre)

model_svm_test <- predict(model_svm, test_set[feats])
confusionMatrix(model_svm_test, test_set$music_genre)

model_svm_test_prob <- predict(model_svm,select(test_set,all_of(feats)), probability = TRUE)
model_svm_prob <- attr(model_svm_test_prob, "probabilities")

model_svm_prob

roc_multi <- multiclass.roc(test_set$music_genre, model_svm_prob)
auc(roc_multi)

multiclass_roc_plot(test_set,model_svm_prob)

var(as.numeric(model_svm_test), as.numeric(test_set$music_genre))

bias(as.numeric(model_svm_test), as.numeric(test_set$music_genre))

set.seed(1111)
model_dtr <- rpart(music_genre ~ ., data = train_set)

rpart.plot(model_dtr, 
           type = 5, 
           extra = 104,
           box.palette = list(purple = "#490B32",
               red = "#9A031E",
               orange = '#FB8B24',
               dark_blue = "#0F4C5C",
               blue = "#5DA9E9",
               grey = '#66717E'),
           leaf.round = 0,
           fallen.leaves = FALSE, 
           branch = 0.3, 
           under = TRUE,
           under.col = 'grey40',
           main = 'Genre Decision Tree',
           tweak = 1.2)
model_dt<- ctree(music_genre ~ ., train_set)

decision_tree <- predict(model_dt, newdata = train_set)
confusionMatrix(decision_tree, as.factor(train_set$music_genre))

predict_model<-predict(model_dt, test_set)
confusionMatrix(predict_model, as.factor(test_set$music_genre))

model_dt_prob <- predict(model_dtr,select(test_set,all_of(feats)), type="prob")
roc_dtr <- multiclass.roc(test_set$music_genre, model_dt_prob) 
auc(roc_dtr)

multiclass_roc_plot(test_set, model_dt_prob)

var(as.numeric(predict_model), as.numeric(test_set$music_genre))

bias(as.numeric(predict_model), as.numeric(test_set$music_genre))

train_knn <- train_set
test_knn <- test_set

#Convert all columns to numeric
index <- 1:ncol(train_knn)

train_knn[ , index] <- lapply(train_knn[ , index], as.numeric)
test_knn[ , index] <- lapply(test_knn[ , index], as.numeric)


model_knn <- knn(train_knn, test_knn, cl = train_knn$music_genre, k = 211)

confusionMatrix(model_knn, as.factor(test_knn$music_genre))

var(as.numeric(model_knn), as.numeric(test_knn$music_genre))

bias(as.numeric(model_knn), as.numeric(test_knn$music_genre))

library(naivebayes)

model <- naive_bayes(music_genre ~ ., data = train_set, usekernel = T) 
model 
plot(model) 

p <- predict(model, train_set, type = 'prob')
p

p_c <- predict(model, train_set, type = 'class')

confusionMatrix(p_c, train_set$music_genre)

p_ct <- predict(model, test_set, type = 'class')
confusionMatrix(p_ct, test_set$music_genre)

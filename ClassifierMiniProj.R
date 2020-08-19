# Machine Learning clssifier mini project
# Shulei Yang
library(formattable)
library(MASS)
library(ElemStatLearn)
library(e1071)
library(kernlab)
library(randomForest)
library(gbm)

train = read.table("traindata.txt")
test = read.table("testdata.txt")

#================= Exploratory Data Analysis ==================#
print('Exploratory Data Analysis: ')
print(paste(' -The proportion of spam emails in the train dataset is ', percent(table(train$V58)['1']/nrow(train))))
print(paste(' -The proportion of spam emails in the test dataset is ', percent(table(test$V58)['1']/nrow(test))))

continuous_count = 0
for(var in colnames(train[ ,-58])){
  if(typeof(train[ , var]) == 'double'){
    continuous_count = continuous_count + 1
  }
}
print(paste(' -There are ', continuous_count, ' continuous variables in the dataset'))
print(paste(' -There are ', ncol(train)-1-continuous_count, ' catagorical variables in the dataset'))

#==================== a) LDA & QDA V55-V57 =====================#
a.lda <- lda(train[ ,55:57], train[ ,58])
a.lda.pred <- predict(a.lda, test[ ,55:57])$class
a.lda.result = table(a.lda.pred, test$V58)
#a.lda.pred   0   1
#         0 913 480
#         1  28 115
print('LDA results for V55-V57: ')
print(paste(' -Accuracy of the model is ', percent((a.lda.result[1,1] + a.lda.result[2,2])/sum(a.lda.result))))
print(paste(' -Sensitivity of the model is ', percent((a.lda.result[2,2])/sum(a.lda.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((a.lda.result[1,1])/sum(a.lda.result[ ,1]))))

a.qda <- qda(train[ ,55:57], train[ ,58])
a.qda.pred <- predict(a.qda, test[ ,55:57])$class
a.qda.result = table(a.qda.pred, test$V58)
#a.qda.pred   0   1
#         0 904 452
#         1  37 143
print('QDA results for V55-V57: ')
print(paste(' -Accuracy of the model is ', percent((a.qda.result[1,1] + a.qda.result[2,2])/sum(a.qda.result))))
print(paste(' -Sensitivity of the model is ', percent((a.qda.result[2,2])/sum(a.qda.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((a.qda.result[1,1])/sum(a.qda.result[ ,1]))))


#==================== b) LDA & QDA V1-V57 =====================#
b.lda <- lda(train[ ,1:57], train[ ,58])
b.lda.pred <- predict(b.lda, test[ ,1:57])$class
b.lda.result = table(b.lda.pred, test$V58)
#b.lda.pred   0   1
#         0 888 135
#         1  53 460
print('LDA results for V1-V57: ')
print(paste(' -Accuracy of the model is ', percent((b.lda.result[1,1] + b.lda.result[2,2])/sum(b.lda.result))))
print(paste(' -Sensitivity of the model is ', percent((b.lda.result[2,2])/sum(b.lda.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((b.lda.result[1,1])/sum(b.lda.result[ ,1]))))

b.qda <- qda(train[ ,1:57], train[ ,58])
b.qda.pred <- predict(b.qda, test[ ,1:57])$class
b.qda.result = table(b.qda.pred, test$V58)
#b.qda.pred   0   1
#         0 706  33
#         1 235 562
print('QDA results for V1-V57: ')
print(paste(' -Accuracy of the model is ', percent((b.qda.result[1,1] + b.qda.result[2,2])/sum(b.qda.result))))
print(paste(' -Sensitivity of the model is ', percent((b.qda.result[2,2])/sum(b.qda.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((b.qda.result[1,1])/sum(b.qda.result[ ,1]))))


#============== c) Logistic Regression & Linear SVM V1-V57 ==============#
c.lg <- glm(V58 ~ ., data = train, family = binomial)
round(summary(c.lg)$coef, dig=3)
c.lg.pred <- predict(c.lg, test[ ,1:57]) > 0.5
c.lg.result <- table(c.lg.pred, test[ ,58])
#c.lg.pred   0   1
#    FALSE 909  87
#    TRUE   32 508
print('Logistic Regression results for V1-V57: ')
print(paste(' -Accuracy of the model is ', percent((c.lg.result[1,1] + c.lg.result[2,2])/sum(c.lg.result))))
print(paste(' -Sensitivity of the model is ', percent((c.lg.result[2,2])/sum(c.lg.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((c.lg.result[1,1])/sum(c.lg.result[ ,1]))))

c.svm <- svm(V58 ~ ., data = train, type='C-classification', kernel='linear',scale=FALSE, cost = 1)
c.svm.pred <- predict(c.svm, test[ ,1:57])
c.svm.result <- table(c.svm.pred, test[ ,58])
print('Linear SVM results for V1-V57: ')
print(paste(' -Accuracy of the model is ', percent((c.svm.result[1,1] + c.svm.result[2,2])/sum(c.svm.result))))
print(paste(' -Sensitivity of the model is ', percent((c.svm.result[2,2])/sum(c.svm.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((c.svm.result[1,1])/sum(c.svm.result[ ,1]))))


#============== d) Non-Linear SVM V1-V57 ==============#
d.svm <- ksvm(V58 ~ V55 + V56 + V57, data = train, kernel = 'rbfdot')
d.svm.pred <- predict(d.svm, test[ ,55:57]) > 0.5
d.svm.result <- table(d.svm.pred, test[, 58])
print('Non-Linear SVM results for V55-V57: ')
print(paste(' -Accuracy of the model is ', percent((d.svm.result[1,1] + d.svm.result[2,2])/sum(d.svm.result))))
print(paste(' -Sensitivity of the model is ', percent((d.svm.result[2,2])/sum(d.svm.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((d.svm.result[1,1])/sum(d.svm.result[ ,1]))))



#================ e) random prediction ================#
test$e.rand = rbinom(nrow(test), 1, 0.5)
e.svm.result <- table(test$e.rand, test[, 58])
print('[Random Prediction] Non-Linear SVM results for V55-V57: ')
print(paste(' -Accuracy of the model is ', percent((e.svm.result[1,1] + e.svm.result[2,2])/sum(e.svm.result))))
print(paste(' -Sensitivity of the model is ', percent((e.svm.result[2,2])/sum(e.svm.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((e.svm.result[1,1])/sum(e.svm.result[ ,1]))))



#=========== f) random prediction sequence of prob ==========#
f.probs = seq(0.1, 0.9, 0.1)
f.result = data.frame(matrix(nrow = length(f.probs)+2, ncol = 2))
colnames(f.result) <- c('Sensitivity', 'Specificity')
row.names(f.result) <- sapply(c(0, f.probs, 1), function(x) paste('prob=', x, sep = ''))
f.result[1, ] = c(0, 1)
f.result[11, ] = c(1, 0)
for(prob in f.probs){
  rand.prob <- rbinom(nrow(test), 1, prob)
  table <- table(rand.prob, test[ , 58])
  f.result[prob*10+1, 'Sensitivity'] = percent((table[2,2])/sum(table[ ,2]))
  f.result[prob*10+1, 'Specificity'] = percent((table[1,1])/sum(table[ ,1]))
}
plot(x = f.result$Specificity, y = f.result$Sensitivity, main = 'Sensitivity and Specificity for probs in 0 to 1');
f.result$sum = apply(f.result, 1, sum)
# f.result ->
#       Sensitivity Specificity       sum
#prob=0    0.00000000  1.00000000 1.0000000
#prob=0.1  0.09411765  0.88947928 0.9835969
#prob=0.2  0.22689076  0.78852285 1.0154136
#prob=0.3  0.30588235  0.71732200 1.0232044
#prob=0.4  0.45042017  0.62167906 1.0720992
#prob=0.5  0.50252101  0.49415515 0.9966762
#prob=0.6  0.58487395  0.41870351 1.0035775
#prob=0.7  0.68907563  0.31243358 1.0015092
#prob=0.8  0.79327731  0.20191286 0.9951902
#prob=0.9  0.88739496  0.09776833 0.9851633
#prob=1    1.00000000  0.00000000 1.0000000
#whtn prob = 0.4, the sum is the largest



#=============== g) random forest ==============#
g.rf.fit <- randomForest(as.factor(V58) ~ ., data = train, ntree = 500, mtry = 7, nodesize = 300, importance = TRUE)
g.rf.pred <- predict(g.rf.fit, test[ ,1:57])
g.rf.result <- table(g.rf.pred, test$V58)
print('Random Forest results for V1-V57: ')
print(paste(' -Accuracy of the model is ', percent((g.rf.result[1,1] + g.rf.result[2,2])/sum(g.rf.result))))
print(paste(' -Sensitivity of the model is ', percent((g.rf.result[2,2])/sum(g.rf.result[ ,2]))))
print(paste(' -specificity of the model is ', percent((g.rf.result[1,1])/sum(g.rf.result[ ,1]))))



#==== h) random forest MeanDecreaseGini barplot ====#
barplot(importance(g.rf.fit)[,4])



#========================= h) boosting ===============================#
shrink <- seq(0.01,0.1,0.01)
size=c(100, 200, 500, 1000, 2000)

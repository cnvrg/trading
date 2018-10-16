# Install dependencies
install.packages('caret', dependencies = TRUE)
install.packages('ggplot2', dependencies = TRUE)
install.packages('gridExtra', dependencies = TRUE)

# Load our data set.
data(iris);

# Load libraries.
library(ggplot2);
library(caret);
library(gridExtra);

set.seed(100);

# Inspect our data set.
names(iris);
head(iris);

# Plot the data.
qp0 = qplot(Petal.Width, Petal.Length, color = Species, data = iris, main = "Full Iris data set (100%)");

iris_k <- kmeans(iris[, c("Petal.Length", "Petal.Width")], 3, nstart = 20);

print(iris_k);
table(iris_k$cluster, iris$Species);

# Plot unsupervised clustering.
iris$cluster <- as.factor(iris_k$cluster);
qp1 <- qplot(Petal.Width, Petal.Length, color = cluster, data = iris, main = "K means clustered, k = 3");

grid.arrange(qp0, qp1, ncol = 2, nrow = 1);

iris$cluster <- NULL;

set.seed(100);

# Split our data into a training (70%) and test (30%) set.
training_split <- createDataPartition(y = iris$Species, p = 0.70, list = FALSE);
training_set <- iris[training_split,];
testing_set <- iris[-training_split,];

# Plot our training data
qp2 <- qplot(Petal.Width, Petal.Length, color = Species, data = training_set, main = "Training data set (70%)");

# Plot our test data
qp3 <- qplot(Petal.Width, Petal.Length, color = Species, data = testing_set, main = "Testing data set (30%)");

# Train our model.
model_fit <- train(Species ~ ., method = "rpart", data = training_set, trControl = trainControl(method = 'cv', number = 10, classProbs = TRUE));
# Using all features.

print(model_fit);

# Classify from our reserved test set.
testing_set_predict <- predict(model_fit, newdata = testing_set[, -5]);
# Remove the Species field to prove we are not cheating.

# Verifying our model from the classifications.
table(testing_set_predict, testing_set$Species);

testing_set$Correct <- (testing_set_predict == testing_set$Species);
accuracy <- length(testing_set$Correct[testing_set$Correct == TRUE]) / length(testing_set$Correct);
paste("TrainAccuracy:", accuracy);

# Plot the classification results.
qp4 <- qplot(Petal.Width, Petal.Length, color = Correct, data = testing_set, main = "Test set classification results");

grid.arrange(qp0, qp2, qp3, qp4, ncol = 2, nrow = 2);

# Perform principal component analysis (by hand).
analysis <- prcomp(training_set[, -5], scale. = TRUE);
print(analysis$sdev);

variance_vector <- analysis$sdev ^ 2;
print(variance_vector);

relative_variance <- variance_vector / sum(variance_vector);
cumsum(relative_variance);
# Looking at the results, only 3 principal components needed to capture 99% of the variance.

# Using the Caret package to confirm results.
preProcess(training_set[, -5], method = "pca", thresh = 0.99);

# Generate RoC curves.

testing_set_predict <- predict(model_fit, newdata = testing_set[, -5], "prob");

roc_setosa <- data.frame(testing_set_predict$setosa, testing_set$Species == 'setosa');
colnames(roc_setosa) <- c("predict", "label");
roc_versicolor <- data.frame(testing_set_predict$versicolor, testing_set$Species == 'versicolor');
colnames(roc_versicolor) <- c("predict", "label");
roc_virginica <- data.frame(testing_set_predict$virginica, testing_set$Species == 'virginica');
colnames(roc_virginica) <- c("predict", "label");

library(ROCR);

pred_setosa <- prediction(roc_setosa$predict, roc_setosa$label);
perf_setosa <- performance(pred_setosa, "tpr", "fpr");
pred_versicolor <- prediction(roc_versicolor$predict, roc_versicolor$label);
perf_versicolor <- performance(pred_versicolor, "tpr", "fpr");
pred_virginica <- prediction(roc_virginica$predict, roc_virginica$label);
perf_virginica <- performance(pred_virginica, "tpr", "fpr");

qproc1 <- qplot(perf_setosa@x.values[[1]], perf_setosa@y.values[[1]], xlab = "FP Rate", ylab = "TP Rate", main = "Setosa", geom = "path");
qproc2 <- qplot(perf_versicolor@x.values[[1]], perf_versicolor@y.values[[1]], xlab = "FP Rate", ylab = "TP Rate", main = "Versicolor", geom = "path");
qproc3 <- qplot(perf_virginica@x.values[[1]], perf_virginica@y.values[[1]], xlab = "FP Rate", ylab = "TP Rate", main = "Virginica", geom = "path");
grid.arrange(qproc1, qproc2, qproc3, ncol = 2, nrow = 2);

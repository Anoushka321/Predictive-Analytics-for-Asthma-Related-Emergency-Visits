data <- read.csv("/Users/anoushkasingh/Documents/Sem 2/Thurs-Data Mining/Project/Singh_Anoushka_&_Munawar_Nida_IntermediateReport/preprocessed_data.csv", header = TRUE)
head(data)


#--------
#--------
#TRAIN/TEST
#--------
#--------

library(caTools)
set.seed(123)

# Split the data into training (67%) and testing (33%) sets
split <- sample.split(data$Class, SplitRatio = 0.67)
training_data <- subset(data, split == TRUE)
testing_data <- subset(data, split == FALSE)
cat("Training set size: ", nrow(training_data), "\n")
cat("Testing set size: ", nrow(testing_data), "\n")



train_file_path <- "/Users/anoushkasingh/Documents/Sem 2/Thurs-Data Mining/Project/Singh_Anoushka_&_Munawar_Nida/initial_train.csv"
test_file_path <- "/Users/anoushkasingh/Documents/Sem 2/Thurs-Data Mining/Project/Singh_Anoushka_&_Munawar_Nida/initial_test.csv"

# Save the training and testing datasets as CSV files
write.csv(training_data, file = train_file_path, row.names = FALSE)
write.csv(testing_data, file = test_file_path, row.names = FALSE)
cat("Initial training and testing datasets have been saved as initial_train.csv and initial_test.csv.\n")


#--------
#--------
#BALANCING DATASET
#--------
#--------

# Find the indices of the majority (Class == "N") and minority class (Class == "Y")
zero_indices <- which(training_data$Class == "N")  
one_indices <- which(training_data$Class == "Y")   

# Count the number of indices for each class
class_counts <- c(length(zero_indices), length(one_indices))
class_names <- c("N", "Y")

data_vis <- data.frame(Class = class_names, Count = class_counts)

ggplot(data_vis, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity") +
  labs(title = "Distribution of Indices by Class",
       x = "Class", y = "Count") +
  theme_minimal()

# Undersampling[B1]: randomly sample the majority class (zeros) to match the number of the minority class (ones)
z_indices <- sample(zero_indices, length(one_indices), replace = FALSE)

# Bootstrap resampling[B2] of both classes
bs_zero_indices <- sample(zero_indices, length(one_indices), replace = TRUE)
bs_one_indices <- sample(one_indices, length(one_indices), replace = TRUE)

# Undersampled training set
train_undersampled <- rbind(training_data[z_indices, ], training_data[one_indices, ])
cat("Undersampled training set class distribution:\n")
table(train_undersampled$Class)

# Bootstrap resampled training set
train_bs <- rbind(training_data[bs_zero_indices, ], training_data[bs_one_indices, ])
cat("Bootstrap resampled training set class distribution:\n")
table(train_bs$Class)

# Check the class distribution in the testing set
cat("Testing set class distribution:\n")
table(testing_data$Class)


dim(train_undersampled) #------------------BALANCED TRAINING SET-1 [B1]
dim(train_bs) #----------------------------BALANCED TRAINING SET-2 [B2]





#--------
#--------
#FEATURE SELECTION
#--------
#--------



#1 - LASSO (F1)


library(glmnet)
library(caret)


train_undersampled$Class <- as.factor(train_undersampled$Class)
x_train <- model.matrix(Class ~ . - 1, data = train_undersampled)
y_train <- train_undersampled$Class

# Cross-validation for LASSO
set.seed(123)
cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", type.measure = "class")
best_lambda <- cv_fit$lambda.min # Get best lambda

# Train final LASSO model
final_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, family = "binomial")
coef_values <- as.matrix(coef(final_model)) # Get coefficients

# Exclude intercept and filter for important features
important_features <- coef_values[-1, ]  
important_features <- important_features[important_features != 0]  
f1 <- names(important_features)  
cat("Important features:\n")
print(f1)

#+-+-+-+-+-+-+-+-+-+-+-
#+#+-+-+-+-+-+-+-+-+-+-+-
#2 - PCA (F2)


library(ggplot2)
library(dplyr)

# Prepare the data for PCA (removing the Class column)
pca_data <- train_undersampled[, -which(names(train_undersampled) == "Class")]
pca_data <- pca_data[, sapply(pca_data, var) != 0]
pca_data_scaled <- scale(pca_data) # Standardize the data
pca_result <- prcomp(pca_data_scaled, center = TRUE, scale. = TRUE)

# Create a data frame for loading scores of the first principal component
loading_scores_df <- data.frame(
  Feature = rownames(pca_result$rotation),
  Score = pca_result$rotation[, 1]
)

# Sort all features by absolute loading score in descending order and store in 'f2'
f2 <- loading_scores_df %>%
  arrange(desc(abs(Score))) %>%
  pull(Feature)


cat("Features from PCA stored in f2 (ordered by loading score):\n")
print(f2)

ggplot(loading_scores_df, aes(x = reorder(Feature, Score), y = Score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Features Ordered by PCA Loading Score", x = "Features", y = "Loading Score") +
  theme_minimal()

#+-+-+-+-+-+-+-+-+-+-+-
#+#+-+-+-+-+-+-+-+-+-+-+-
#3 - INFORMATION GAIN (F3)


library(caret)

# Train a logistic regression model with caret
model <- train(Class ~ ., data = train_undersampled, method = "glm", family = "binomial")
importance <- varImp(model)

# Convert importance into a data frame and sort by importance
importance_df <- as.data.frame(importance$importance)
top_features <- rownames(importance_df)[order(-importance_df[, 1])] 
f3 <- top_features
cat("Top features based on information gain stored in f3:\n")
print(f3)



#+-+-+-+-+-+-+-+-+-+-+-
#+-+-+-+-+-+-+-+-+-+-+-

# ML ALGORITHM
#------------------------------------
#------------------------------------

#FOR B1,F1- UNDERSAMPLED AND LASSO

#------------------------------------
#------------------------------------
#1- RANDOM FOREST [B1,F1]
#------------------------------------
#------------------------------------

library(randomForest)
library(pROC)
library(psych)

#f1 contains the names of the important features from LASSO
selected_features <- f1

x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]

# Train the Random Forest model
set.seed(123)
rf_model <- randomForest(x = x_train_undersampled, y = y_train_undersampled, ntree = 500)
predictions <- predict(rf_model, newdata = x_test)

calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total
  
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class

# Train the Random Forest model
set.seed(123)
rf_model <- randomForest(x = x_train_undersampled, y = y_train_undersampled, ntree = 500)
predictions <- predict(rf_model, newdata = x_test)

# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)y
cat("Accuracy:", round(accuracy, 4), "\n")

roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for Random Forest", 
     col = "red", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightpink",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#2- DECISION TREE [B1,F1]
#------------------------------------
#------------------------------------


library(rpart)
library(caret)

#f1 contains the names of the important features from LASSO
selected_features <- f1

x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
train_data <- data.frame(Class = y_train_undersampled, x_train_undersampled)
x_test <- testing_data[, selected_features]

# Train the Decision Tree model
set.seed(123)
dt_model <- rpart(Class ~ ., data = train_data, method = "class")
predictions <- predict(dt_model, newdata = x_test, type = "class")


calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total
  
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Create and print metrics table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class


# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")


# Calculate the ROC curve for the model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for Decision Tree", 
     col = "blue", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightblue",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#3- SVM [B1,F1]
#------------------------------------
#------------------------------------


library(e1071) 
library(caret)  

#f1 contains the names of the important features from LASSO
selected_features <- f1

x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
train_data <- data.frame(Class = y_train_undersampled, x_train_undersampled)
x_test <- testing_data[, selected_features]

# Train the SVM model
set.seed(123)
svm_model <- svm(Class ~ ., data = train_data, kernel = "linear")
predictions <- predict(svm_model, newdata = x_test)


calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total
  
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Create and print metrics table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class


# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")


# Calculate the ROC curve for the model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for SVM", 
     col = "darkgreen", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightgreen",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#------------------------------------
#------------------------------------
#4- GBM [B1,F1]
#------------------------------------
#------------------------------------

library(caret)    
library(gbm)      

# Use the important features from LASSO
selected_features <- f1  


x_train_undersampled <- train_undersampled[, selected_features, drop = FALSE]
y_train_undersampled <- as.factor(train_undersampled$Class)  

x_test <- testing_data[, selected_features, drop = FALSE]
y_test <- as.factor(testing_data$Class)
train_data <- data.frame(x_train_undersampled, Class = y_train_undersampled)
test_data <- data.frame(x_test, Class = y_test)

# Define the training control
train_control <- trainControl(
  method = "cv",                 
  number = 5,                    
  classProbs = TRUE,            
  summaryFunction = twoClassSummary  
)

# Train the GBM model
set.seed(123)
gbm_model <- train(
  Class ~ ., data = train_data,
  method = "gbm",
  trControl = train_control,
  metric = "ROC",               
  verbose = FALSE,
  tuneLength = 5                 
)


predictions <- predict(gbm_model, newdata = test_data)
prob_predictions <- predict(gbm_model, newdata = test_data, type = "prob")


calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total
  
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Create and print metrics table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class


# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")


# Calculate the ROC curve for the model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for GBM", 
     col = "orange", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightyellow",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#5- KNN [B1,F1]
#------------------------------------
#------------------------------------


library(caret)
library(class)  

#f1 contains the names of the important features from LASSO
selected_features <- f1

x_train <- train_undersampled[, selected_features]
y_train <- as.factor(train_undersampled$Class)  # Ensure 'Class' is a factor
x_test <- testing_data[, selected_features]
y_test <- as.factor(testing_data$Class)

# Train KNN model
set.seed(123)
knn_predictions <- knn(train = x_train, test = x_test, cl = y_train, k = 5)  # Set k as needed


calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total
  
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Create and print metrics table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class


# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")


# Calculate the ROC curve for the model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for KNN", 
     col = "purple", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lavender",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#6- XGBoost [B1,F1]
#------------------------------------
#------------------------------------


library(xgboost)
library(caret)

# Set target and features based on selected LASSO features
selected_features <- f1
x_train <- as.matrix(train_undersampled[, selected_features])
y_train <- as.numeric(train_undersampled$Class == "Y")  # Convert "Y" to 1, "N" to 0

x_test <- as.matrix(testing_data[, selected_features])
y_test <- as.numeric(testing_data$Class == "Y")

# Train XGBoost model
set.seed(123)
xgb_train <- xgb.DMatrix(data = x_train, label = y_train)
xgb_test <- xgb.DMatrix(data = x_test, label = y_test)

# Parameters for XGBoost
params <- list(
  objective = "binary:logistic",
  eval_metric = "logloss",
  max_depth = 6,
  eta = 0.3
)

# Train the XGBoost model
xgb_model <- xgb.train(
  params = params,
  data = xgb_train,
  nrounds = 100,
  verbose = 0
)


pred_prob <- predict(xgb_model, newdata = xgb_test)
predictions <- ifelse(pred_prob > 0.5, "Y", "N")
predictions <- factor(predictions, levels = c("N", "Y"))
y_test <- factor(ifelse(y_test == 1, "Y", "N"), levels = c("N", "Y"))


calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total
  
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Create and print metrics table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class


# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")


# Calculate the ROC curve for the model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for XGBoost", 
     col = "#009999", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "#CCFFFF",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#------------------------------------
#------------------------------------

#FOR B1,F2- UNDERSAMPLED AND PCA

#------------------------------------
#------------------------------------
#1- RANDOM FOREST [B1,F2]
#------------------------------------
#------------------------------------


library(randomForest)
library(caret)

# Filter undersampled data (b1) to keep only features in f2 (PCA-selected features)
b1 <- train_undersampled[, c(f2, "Class")]
b1$Class <- as.factor(b1$Class)

# Split features and target variable
x_train <- b1[, -which(names(b1) == "Class")]
y_train <- b1$Class

# Train the Random Forest model
set.seed(123)
rf_model <- randomForest(x = x_train, y = y_train, ntree = 100)

# Make predictions on the testing set (assuming `testing_data` exists)
test_data <- testing_data[, c(f2, "Class")]
test_data$Class <- as.factor(test_data$Class)
x_test <- test_data[, -which(names(test_data) == "Class")]
y_test <- test_data$Class
predictions <- predict(rf_model, newdata = x_test)


calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total

  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Create and print metrics table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class


# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")


# Calculate the ROC curve for the model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for Random Forest", 
     col = "red", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightpink",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#------------------------------------
#------------------------------------
#2- DECISION TREE [B1,F2]
#------------------------------------
#------------------------------------


library(rpart)
library(caret)

# Filter undersampled data (b1) to keep only features in f2 (PCA-selected features)
b1 <- train_undersampled[, c(f2, "Class")]

# Convert 'Class' to a factor for Decision Tree
b1$Class <- as.factor(b1$Class)

# Split features and target variable for training
x_train <- b1[, -which(names(b1) == "Class")]
y_train <- b1$Class

# Train the Decision Tree model
set.seed(123)
dt_model <- rpart(Class ~ ., data = b1, method = "class")

# Prepare the testing data
test_data <- testing_data[, c(f2, "Class")]
test_data$Class <- as.factor(test_data$Class)
x_test <- test_data[, -which(names(test_data) == "Class")]
y_test <- test_data$Class
predictions <- predict(dt_model, newdata = x_test, type = "class")




calculate_metrics <- function(conf_matrix, actual, predicted) {
  
  # Extract values from the confusion matrix
  TP <- conf_matrix["Y", "Y"]
  TN <- conf_matrix["N", "N"]
  FP <- conf_matrix["Y", "N"]
  FN <- conf_matrix["N", "Y"]
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- TN / (TN + FP)  # TPR for Class N
  tpr_y <- TP / (TP + FN)  # TPR for Class Y
  fpr_n <- FP / (FP + TN)  # FPR for Class N
  fpr_y <- FN / (FN + TP)  # FPR for Class Y
  
  precision_y <- TP / (TP + FP)  # Precision for Class Y
  precision_n <- TN / (TN + FN)  # Precision for Class N
  
  recall_y <- tpr_y  # Recall for Class Y, same as TPR
  recall_n <- tpr_n  # Recall for Class N, same as TPR
  
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  # MCC Calculation with floating-point conversion to avoid overflow
  mcc <- ((as.numeric(TP) * as.numeric(TN)) - (as.numeric(FP) * as.numeric(FN))) / 
    sqrt((as.numeric(TP) + as.numeric(FP)) * (as.numeric(TP) + as.numeric(FN)) * 
           (as.numeric(TN) + as.numeric(FP)) * (as.numeric(TN) + as.numeric(FN)))
  
  # Kappa Statistic
  kappa_stat <- cohen.kappa(conf_matrix)$kappa
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Weighted Averages
  total <- sum(conf_matrix)
  weighted_avg_tpr <- (tpr_n * sum(actual == "N") + tpr_y * sum(actual == "Y")) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == "N") + fpr_y * sum(actual == "Y")) / total
  weighted_avg_precision <- (precision_n * sum(actual == "N") + precision_y * sum(actual == "Y")) / total
  weighted_avg_recall <- (recall_n * sum(actual == "N") + recall_y * sum(actual == "Y")) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == "N") + f1_score_y * sum(actual == "Y")) / total
  
  cat("Confusion Matrix:\n")
  print(conf_matrix)
  
  # Create and print metrics table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Selected features and data partitioning
selected_features <- f1
x_train_undersampled <- train_undersampled[, selected_features]
y_train_undersampled <- train_undersampled$Class
x_test <- testing_data[, selected_features]
y_test <- testing_data$Class


# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")


# Calculate the ROC curve for the model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for Decision Tree", 
     col = "blue", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightblue",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#3- SVM [B1,F2]
#------------------------------------
#------------------------------------

library(e1071)
library(caret)
library(pROC)

# Filter undersampled data (b1) to keep only features in f2 (PCA-selected features)
b1 <- train_undersampled[, c(f2, "Class")]
b1$Class <- as.factor(b1$Class)

# Split features and target variable for training
x_train <- b1[, -which(names(b1) == "Class")]
y_train <- b1$Class

# Train the SVM model
set.seed(123)
svm_model <- svm(Class ~ ., data = b1, kernel = "linear")  # You can choose other kernels as well

# Prepare the testing data
test_data <- testing_data[, c(f2, "Class")]
test_data$Class <- as.factor(test_data$Class)
x_test <- test_data[, -which(names(test_data) == "Class")]
y_test <- test_data$Class

# Make predictions
predictions <- predict(svm_model, x_test)

# Confusion Matrix
confusion_matrix <- table(Predicted = predictions, Actual = y_test)
calculate_metrics(confusion_matrix, y_test, predictions)

# Calculate accuracy
accuracy <- sum(diag(confusion_matrix)) / sum(confusion_matrix)
cat("Accuracy:", round(accuracy, 4), "\n")

# Calculate the ROC curve for the SVM model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for SVM", 
     col = "darkgreen", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightgreen",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#4- XGBOOST [B1,F2]
#------------------------------------
#------------------------------------


library(xgboost)
library(caret)
library(pROC)  

# Assuming train_undersampled and testing_data are already defined
# Filter undersampled data (b1) to keep only features in f2 (PCA-selected features)
b1 <- train_undersampled[, c(f2, "Class")]
b1$Class <- as.factor(b1$Class)

# Prepare the data for XGBoost
x_train <- as.matrix(b1[, -which(names(b1) == "Class")])
y_train <- as.numeric(b1$Class) - 1  # Convert to numeric (0 and 1)

# Set parameters for the XGBoost model
params <- list(
  objective = "binary:logistic",  # For binary classification
  eval_metric = "logloss",         # Evaluation metric
  nthread = 2,                     # Number of threads
  max_depth = 6,                   # Maximum tree depth
  eta = 0.3,                       # Learning rate
  gamma = 0,                       # Minimum loss reduction required to make a further partition
  subsample = 0.7                   # Subsample ratio of the training instances
)

# Train the XGBoost model
set.seed(123)
xgb_model <- xgboost(data = x_train, label = y_train, params = params, nrounds = 100, verbose = 0)

# Prepare the testing data
test_data <- testing_data[, c(f2, "Class")]
test_data$Class <- as.factor(test_data$Class)
x_test <- as.matrix(test_data[, -which(names(test_data) == "Class")])
y_test <- as.numeric(test_data$Class) - 1  # Convert to numeric (0 and 1)

# Generate predictions
predictions_prob <- predict(xgb_model, newdata = x_test)
predictions <- ifelse(predictions_prob > 0.5, 1, 0)  # Thresholding at 0.5

# Function to calculate metrics
calculate_metrics <- function(actual, predicted) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Calculate Accuracy
  accuracy <- sum(diag(cm)) / sum(cm)  # TP + TN / Total
  cat("Accuracy:", accuracy, "\n")
  
  # Calculate metrics for Class N and Class Y
  tpr_n <- cm[1, 1] / (cm[1, 1] + cm[1, 2])  # True Positive Rate for Class N
  fpr_n <- cm[1, 2] / (cm[1, 1] + cm[2, 2])  # False Positive Rate for Class N
  precision_n <- cm[1, 1] / (cm[1, 1] + cm[2, 1])  # Precision for Class N
  recall_n <- tpr_n  # Recall for Class N (same as TPR)
  f1_score_n <- (2 * precision_n * recall_n) / (precision_n + recall_n)  # F1 Score for Class N
  
  tpr_y <- cm[2, 2] / (cm[2, 2] + cm[2, 1])  # True Positive Rate for Class Y
  fpr_y <- cm[2, 1] / (cm[2, 2] + cm[1, 1])  # False Positive Rate for Class Y
  precision_y <- cm[2, 2] / (cm[2, 2] + cm[1, 2])  # Precision for Class Y
  recall_y <- tpr_y  # Recall for Class Y (same as TPR)
  f1_score_y <- (2 * precision_y * recall_y) / (precision_y + recall_y)  # F1 Score for Class Y
  
  # Calculate MCC (Matthews correlation coefficient)
  mcc <- ((cm[1, 1] * cm[2, 2]) - (cm[1, 2] * cm[2, 1])) / 
    sqrt((cm[1, 1] + cm[1, 2]) * (cm[1, 1] + cm[2, 1]) * (cm[2, 2] + cm[1, 2]) * (cm[2, 2] + cm[2, 1]))
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Calculate ROC Area
  roc_obj <- roc(actual, predictions_prob)  # Use predicted probabilities for ROC
  roc_area <- auc(roc_obj)
  
  # Calculate weighted averages
  total <- sum(cm)
  weighted_avg_tpr <- (tpr_n * sum(actual == 0) + tpr_y * sum(actual == 1)) / total
  weighted_avg_fpr <- (fpr_n * sum(actual == 0) + fpr_y * sum(actual == 1)) / total
  weighted_avg_precision <- (precision_n * sum(actual == 0) + precision_y * sum(actual == 1)) / total
  weighted_avg_recall <- (recall_n * sum(actual == 0) + recall_y * sum(actual == 1)) / total
  weighted_avg_f1 <- (f1_score_n * sum(actual == 0) + f1_score_y * sum(actual == 1)) / total
  
  # Create a performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_n, fpr_n, precision_n, recall_n, f1_score_n, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_y, fpr_y, precision_y, recall_y, f1_score_y, NA, NA, kappa_stat),  # ROC and MCC not applicable for Class Y
    Weighted_Avg = c(weighted_avg_tpr, weighted_avg_fpr, weighted_avg_precision, weighted_avg_recall, weighted_avg_f1, roc_area, mcc, kappa_stat)
  )
  
  cat("Confusion Matrix:\n")
  print(cm)
  cat("\nPerformance Measures:\n")
  print(metrics)
}

# Call the function to see results
calculate_metrics(y_test, predictions)



# Calculate the ROC curve for the SVM model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for XGBoost", 
     col = "#009999", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "#CCFFFF",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")





#------------------------------------
#------------------------------------
#5- GBM [B1,F2]
#------------------------------------
#------------------------------------

library(gbm)
library(caret)

# Filter the undersampled data (b1) to keep only features in f2 (PCA-selected features)
b1 <- train_undersampled[, c(f2, "Class")]

# Convert 'Class' to a numeric binary format (0 and 1)
b1$Class <- ifelse(b1$Class == "Y", 1, 0)

# Prepare the data for GBM
x_train <- b1[, -which(names(b1) == "Class")]
y_train <- b1$Class

# Set parameters for the GBM model
set.seed(123)
gbm_model <- gbm(
  formula = Class ~ .,            # Class is the target variable
  distribution = "bernoulli",     # For binary classification
  data = b1,
  n.trees = 1000,                 # Number of trees
  interaction.depth = 3,          # Depth of each tree
  n.minobsinnode = 10,            # Minimum number of observations in the trees
  shrinkage = 0.01,               # Learning rate
  bag.fraction = 0.5,             # Fraction of observations to be used for each tree
  train.fraction = 0.8,           # Fraction of data for training
  verbose = TRUE,                 # Print training progress
  n.cores = 2                     # Number of cores to use for parallel processing
)

# Prepare the testing data
test_data <- testing_data[, c(f2, "Class")]
test_data$Class <- ifelse(test_data$Class == "Y", 1, 0)  # Convert to numeric binary format
predictions_prob <- predict(gbm_model, newdata = test_data, n.trees = 1000, type = "response")
predictions <- ifelse(predictions_prob > 0.5, 1, 0)  # Thresholding at 0.5

# Generate the confusion matrix
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(y_test))
print(conf_matrix$table)

# Function to calculate performance metrics for each class and weighted average
calculate_class_metrics <- function(actual, predicted, predicted_prob) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Calculate metrics for Class Y
  TP_Y <- cm[2, 2]
  FN_Y <- cm[2, 1]
  FP_Y <- cm[1, 2]
  TN_Y <- cm[1, 1]
  
  tpr_Y <- TP_Y / (TP_Y + FN_Y)
  fpr_Y <- FP_Y / (FP_Y + TN_Y)
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y
  FN_N <- FP_Y
  FP_N <- FN_Y
  TN_N <- TP_Y
  
  tpr_N <- TP_N / (TP_N + FN_N)
  fpr_N <- FP_N / (FP_N + TN_N)
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Corrected MCC calculation
  mcc_numerator <- as.numeric((TP_Y * TN_Y) - (FP_Y * FN_Y))
  mcc_denominator <- sqrt(as.numeric((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y)))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # ROC area
  roc_obj <- roc(actual, predicted_prob)
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
}

# Call the function to see the results
calculate_class_metrics(y_test, predictions, predictions_prob)


# Calculate the ROC curve for the SVM model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for GBM", 
     col = "orange", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightyellow",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#------------------------------------
#------------------------------------
#6- KNN [B1,F2]
#------------------------------------
#------------------------------------


library(class)
library(caret)
library(dplyr)
library(pROC)

# Prepare the data for KNN (using the undersampled data)
b1 <- train_undersampled[, c(f2, "Class")]
b1$Class <- as.factor(b1$Class)  # Ensure Class is a factor

# Split data into training and testing sets
set.seed(123)
train_indices <- sample(1:nrow(b1), size = 0.8 * nrow(b1))  # 80% for training
train_data <- b1[train_indices, ]
test_data <- b1[-train_indices, ]

# Prepare training and testing sets for KNN
x_train <- as.matrix(train_data[, -which(names(train_data) == "Class")])  # Features
y_train <- train_data$Class  # Target variable
x_test <- as.matrix(test_data[, -which(names(test_data) == "Class")])  # Features
y_test <- test_data$Class  # Target variable

# Train the KNN model
k_value <- 5  # You can choose an appropriate value for K
predictions <- knn(x_train, x_test, y_train, k = k_value)

# Calculate confusion matrix
conf_matrix <- confusionMatrix(predictions, y_test)
print(conf_matrix)

# Extract TP, TN, FP, FN from confusion matrix for each class
tp_class_y <- conf_matrix$table[2, 2]
tn_class_y <- conf_matrix$table[1, 1]
fp_class_y <- conf_matrix$table[1, 2]
fn_class_y <- conf_matrix$table[2, 1]

tp_class_n <- tn_class_y
tn_class_n <- tp_class_y
fp_class_n <- fn_class_y
fn_class_n <- fp_class_y

# Calculate performance metrics for each class and the weighted average
calculate_metrics <- function(tp, tn, fp, fn) {
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  tpr <- tp / (tp + fn)  # Sensitivity
  fpr <- fp / (fp + tn)  # Fall-out or false positive rate
  precision <- tp / (tp + fp)
  recall <- tpr  # Same as sensitivity
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # MCC calculation
  mcc <- ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  
  list(Accuracy = accuracy, TPR = tpr, FPR = fpr, Precision = precision, Recall = recall,
       F1_Score = f1_score, MCC = mcc)
}

metrics_class_n <- calculate_metrics(tp_class_n, tn_class_n, fp_class_n, fn_class_n)
metrics_class_y <- calculate_metrics(tp_class_y, tn_class_y, fp_class_y, fn_class_y)

# Calculate ROC area for each class and weighted average
roc_obj <- roc(as.numeric(y_test), as.numeric(predictions) - 1)
roc_area <- auc(roc_obj)

# Calculate Kappa statistic
kappa_stat <- conf_matrix$overall["Kappa"]

# Weighted Average
total <- sum(c(tp_class_y, tn_class_y, fp_class_y, fn_class_y))
weighted_avg <- sapply(names(metrics_class_n), function(metric) {
  (metrics_class_n[[metric]] * (tp_class_n + fn_class_n) + metrics_class_y[[metric]] * (tp_class_y + fn_class_y)) / total
})

# Display the performance table
performance_table <- data.frame(
  Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
  Class_N = c(metrics_class_n$TPR, metrics_class_n$FPR, metrics_class_n$Precision, metrics_class_n$Recall, 
              metrics_class_n$F1_Score, roc_area, metrics_class_n$MCC, kappa_stat),
  Class_Y = c(metrics_class_y$TPR, metrics_class_y$FPR, metrics_class_y$Precision, metrics_class_y$Recall, 
              metrics_class_y$F1_Score, roc_area, metrics_class_y$MCC, kappa_stat),
  Weighted_Avg = c(weighted_avg["TPR"], weighted_avg["FPR"], weighted_avg["Precision"], weighted_avg["Recall"],
                   weighted_avg["F1_Score"], roc_area, weighted_avg["MCC"], kappa_stat)
)

print("Confusion Matrix:")
print(conf_matrix)
print("Performance Measures Table:")
print(performance_table)



# Calculate the ROC curve for the SVM model
roc_curve <- roc(y_test, as.numeric(predictions) - 1)  # Convert predictions to numeric if needed
plot(roc_curve, 
     main = "ROC Curve for KNN", 
     col = "purple", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lavender",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#------------------------------------
#------------------------------------

#FOR B1,F3- UNDERSAMPLED AND INFROMATION GAIN

#------------------------------------
#------------------------------------
#1- RANDOM FOREST [B1,F3]
#------------------------------------
#------------------------------------

library(randomForest)
library(caret)
library(pROC)
library(dplyr)

# Assuming you have already defined f3 and train_undersampled as per your previous steps
# Filter the undersampled data (train_undersampled) to keep only features in f3 (top features from information gain)
b1 <- train_undersampled[, c(f3, "Class")]  # Ensure to include Class
b1$Class <- as.factor(b1$Class)

# Prepare the data for Random Forest
x_train <- b1[, -which(names(b1) == "Class")]  # Features
y_train <- b1$Class  # Target variable

set.seed(123)

# Train the Random Forest model
rf_model <- randomForest(x = x_train, y = y_train, ntree = 500)

# Make predictions on the training set
train_predictions <- predict(rf_model, x_train)

# Calculate confusion matrix
conf_matrix <- confusionMatrix(train_predictions, y_train)
print("Confusion Matrix:")
print(conf_matrix)

# Extract TP, TN, FP, FN from confusion matrix for each class
confusion_table <- as.table(conf_matrix)
tp_class_y <- confusion_table[2, 2]  # True Positives for Class Y
tn_class_y <- confusion_table[1, 1]  # True Negatives for Class Y
fp_class_y <- confusion_table[1, 2]  # False Positives for Class Y
fn_class_y <- confusion_table[2, 1]  # False Negatives for Class Y

tp_class_n <- tn_class_y
tn_class_n <- tp_class_y
fp_class_n <- fn_class_y
fn_class_n <- fp_class_y

# Calculate performance metrics for each class and the weighted average
calculate_metrics <- function(tp, tn, fp, fn) {
  accuracy <- (tp + tn) / (tp + tn + fp + fn)
  tpr <- tp / (tp + fn)  # Sensitivity
  fpr <- fp / (fp + tn)  # Fall-out or false positive rate
  precision <- tp / (tp + fp)
  recall <- tpr  # Same as sensitivity
  f1_score <- 2 * (precision * recall) / (precision + recall)
  
  # MCC calculation
  mcc <- ((tp * tn) - (fp * fn)) / sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
  if (is.nan(mcc)) {
    mcc <- NA  # Assign NA if computation fails
  }
  
  list(Accuracy = accuracy, TPR = tpr, FPR = fpr, Precision = precision, Recall = recall,
       F1_Score = f1_score, MCC = mcc)
}

metrics_class_n <- calculate_metrics(tp_class_n, tn_class_n, fp_class_n, fn_class_n)
metrics_class_y <- calculate_metrics(tp_class_y, tn_class_y, fp_class_y, fn_class_y)

# Calculate ROC area for each class and weighted average
roc_obj <- roc(as.numeric(y_train), as.numeric(train_predictions) - 1)
roc_area <- auc(roc_obj)

# Calculate Kappa statistic
kappa_stat <- conf_matrix$overall["Kappa"]

# Weighted Average
total <- sum(c(tp_class_y, tn_class_y, fp_class_y, fn_class_y))
weighted_avg <- sapply(names(metrics_class_n), function(metric) {
  (metrics_class_n[[metric]] * (tp_class_n + fn_class_n) + metrics_class_y[[metric]] * (tp_class_y + fn_class_y)) / total
})

# Display the performance table
performance_table <- data.frame(
  Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
  Class_N = c(metrics_class_n$TPR, metrics_class_n$FPR, metrics_class_n$Precision, metrics_class_n$Recall, 
              metrics_class_n$F1_Score, roc_area, metrics_class_n$MCC, kappa_stat),
  Class_Y = c(metrics_class_y$TPR, metrics_class_y$FPR, metrics_class_y$Precision, metrics_class_y$Recall, 
              metrics_class_y$F1_Score, roc_area, metrics_class_y$MCC, kappa_stat),
  Weighted_Avg = c(weighted_avg["TPR"], weighted_avg["FPR"], weighted_avg["Precision"], weighted_avg["Recall"],
                   weighted_avg["F1_Score"], roc_area, weighted_avg["MCC"], kappa_stat)
)

# Show Accuracy
accuracy_value <- (sum(diag(conf_matrix$table)) / sum(conf_matrix$table)) * 100
cat("Accuracy: ", accuracy_value, "%\n")

print("Performance Measures Table:")
print(performance_table)

# Plot ROC Curve
roc_obj <- roc(as.numeric(y_train), as.numeric(train_predictions) - 1)
plot(roc_obj, 
     main = "ROC Curve for Random Forest Model", 
     col = "red", 
     lwd = 2, 
     legacy.axes = TRUE,
     auc.polygon.col = "pink",  # Color of the shaded area,
     print.auc = TRUE, 
     print.thres = "best",
     threshold = "best")





#------------------------------------
#------------------------------------
#2- DECISION TREE [B1,F3]
#------------------------------------
#------------------------------------


library(rpart)        
library(caret)        

# Filter the undersampled data (train_undersampled) to keep only features in f3
b1 <- train_undersampled[, c(f3, "Class")]  # Ensure to include Class
b1$Class <- as.factor(b1$Class)

# Prepare the data for Decision Tree
x_train <- b1[, -which(names(b1) == "Class")]  # Features
y_train <- b1$Class  # Target variable

set.seed(123)

# Train the Decision Tree model
dt_model <- rpart(Class ~ ., data = b1, method = "class")
train_predictions <- predict(dt_model, x_train, type = "class")
conf_matrix <- confusionMatrix(train_predictions, y_train)
print(conf_matrix)

# Load necessary libraries
library(rpart)
library(caret)
library(pROC)
library(dplyr)

# Filter the undersampled data (train_undersampled) to keep only features in f3
b1 <- train_undersampled[, c(f3, "Class")]  # Ensure to include Class
b1$Class <- as.factor(b1$Class)

# Prepare the data for Decision Tree
x_train <- b1[, -which(names(b1) == "Class")]  # Features
y_train <- b1$Class  # Target variable

set.seed(123)

# Train the Decision Tree model
dt_model <- rpart(Class ~ ., data = b1, method = "class")
train_predictions <- predict(dt_model, x_train, type = "class")

# Confusion Matrix
conf_matrix <- confusionMatrix(train_predictions, y_train)
print(conf_matrix)

# Calculate Performance Measures
confusion_table <- as.table(conf_matrix$table)

TP_N <- confusion_table["N", "N"]  # True Positives for Class N
TN_N <- confusion_table["Y", "Y"]  # True Negatives for Class N
FP_N <- confusion_table["Y", "N"]  # False Positives for Class N
FN_N <- confusion_table["N", "Y"]  # False Negatives for Class N

TP_Y <- confusion_table["Y", "Y"]  # True Positives for Class Y
TN_Y <- confusion_table["N", "N"]  # True Negatives for Class Y
FP_Y <- confusion_table["N", "Y"]  # False Positives for Class Y
FN_Y <- confusion_table["Y", "N"]  # False Negatives for Class Y

# Performance Metrics for Class N
TPR_N <- TP_N / (TP_N + FN_N)
FPR_N <- FP_N / (FP_N + TN_N)
Precision_N <- ifelse((TP_N + FP_N) == 0, 0, TP_N / (TP_N + FP_N))
Recall_N <- TPR_N
Fmeasure_N <- ifelse((Precision_N + Recall_N) == 0, 0, (2 * Precision_N * Recall_N) / (Precision_N + Recall_N))

# Performance Metrics for Class Y
TPR_Y <- TP_Y / (TP_Y + FN_Y)
FPR_Y <- FP_Y / (FP_Y + TN_Y)
Precision_Y <- ifelse((TP_Y + FP_Y) == 0, 0, TP_Y / (TP_Y + FP_Y))
Recall_Y <- TPR_Y
Fmeasure_Y <- ifelse((Precision_Y + Recall_Y) == 0, 0, (2 * Precision_Y * Recall_Y) / (Precision_Y + Recall_Y))

# Calculate MCC
MCC_N <- ifelse((TP_N + FP_N) * (TP_N + FN_N) * (TN_N + FP_N) * (TN_N + FN_N) == 0, 0,
                (TP_N * TN_N - FP_N * FN_N) / sqrt((TP_N + FP_N) * (TP_N + FN_N) * (TN_N + FP_N) * (TN_N + FN_N)))

MCC_Y <- ifelse((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y) == 0, 0,
                (TP_Y * TN_Y - FP_Y * FN_Y) / sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y)))

# ROC Curve and AUC
roc_obj <- roc(y_train, as.numeric(train_predictions), levels = levels(y_train))
roc_area <- auc(roc_obj)

# Weighted Average
weights <- table(y_train) / length(y_train)
TPR_Weighted <- TPR_N * weights["N"] + TPR_Y * weights["Y"]
FPR_Weighted <- FPR_N * weights["N"] + FPR_Y * weights["Y"]
Precision_Weighted <- Precision_N * weights["N"] + Precision_Y * weights["Y"]
Recall_Weighted <- Recall_N * weights["N"] + Recall_Y * weights["Y"]
Fmeasure_Weighted <- Fmeasure_N * weights["N"] + Fmeasure_Y * weights["Y"]
MCC_Weighted <- MCC_N * weights["N"] + MCC_Y * weights["Y"]

# Performance Measure Table
performance_table <- data.frame(
  TPR = c(TPR_N, TPR_Y, TPR_Weighted),
  FPR = c(FPR_N, FPR_Y, FPR_Weighted),
  Precision = c(Precision_N, Precision_Y, Precision_Weighted),
  Recall = c(Recall_N, Recall_Y, Recall_Weighted),
  Fmeasure = c(Fmeasure_N, Fmeasure_Y, Fmeasure_Weighted),
  ROC = c(roc_area, roc_area, roc_area),  # AUC for both classes and weighted average
  MCC = c(MCC_N, MCC_Y, MCC_Weighted),
  Kappa = c(conf_matrix$overall['Kappa'], conf_matrix$overall['Kappa'], conf_matrix$overall['Kappa'])  # Kappa statistic is the same for both classes
)

rownames(performance_table) <- c("Class N", "Class Y", "Weighted Average")
print(performance_table)
accuracy <- conf_matrix$overall['Accuracy']
cat("Accuracy:", accuracy, "\n")
cat("ROC Area (AUC):", roc_area, "\n")

probabilities <- predict(dt_model, x_train, type = "prob")[, "Y"]  # Get probabilities for Class Y

# Create ROC object using the true class labels and predicted probabilities
roc_obj <- roc(y_train, probabilities)
roc_area <- auc(roc_obj)
plot(roc_obj, 
     main = "ROC Curve for Decision Tree", 
     col = "blue", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,      # Shade the area under the curve
     auc.polygon.col = "lightblue",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")




#------------------------------------
#------------------------------------
#3- SVM [B1,F3]
#------------------------------------
#------------------------------------

library(e1071)
library(caret)
library(pROC)

# Filter the undersampled data (train_undersampled) to keep only features in f3
b1 <- train_undersampled[, c(f3, "Class")]  # Ensure to include Class
b1$Class <- as.factor(b1$Class)

# Prepare the data for SVM
x_train <- b1[, -which(names(b1) == "Class")]  # Features
y_train <- b1$Class  # Target variable

set.seed(123)

# Train the SVM model
svm_model <- svm(Class ~ ., data = b1, kernel = "linear", probability = TRUE)

# Predict on the training set
train_predictions <- predict(svm_model, x_train)
conf_matrix <- confusionMatrix(train_predictions, y_train)
print(conf_matrix)

# Calculate True Positives, True Negatives, False Positives, False Negatives
TP_N <- conf_matrix$table[1, 1]  # True Positives for class N
TN_N <- conf_matrix$table[1, 2]  # True Negatives for class N
FP_N <- conf_matrix$table[2, 1]  # False Positives for class N
FN_N <- conf_matrix$table[2, 2]  # False Negatives for class N

TP_Y <- conf_matrix$table[2, 2]  # True Positives for class Y
TN_Y <- conf_matrix$table[2, 1]  # True Negatives for class Y
FP_Y <- conf_matrix$table[1, 2]  # False Positives for class Y
FN_Y <- conf_matrix$table[1, 1]  # False Negatives for class Y

# Metrics Calculation for Class N
TPR_N <- TP_N / (TP_N + FN_N)  # Sensitivity for class N
FPR_N <- FP_N / (FP_N + TN_N)  # Fall-out for class N
Precision_N <- TP_N / (TP_N + FP_N)  # Precision for class N
Recall_N <- TPR_N  # Recall is the same as TPR in this context
F_measure_N <- 2 * (Precision_N * Recall_N) / (Precision_N + Recall_N)  # F-measure for class N

# Metrics Calculation for Class Y
TPR_Y <- TP_Y / (TP_Y + FN_Y)  # Sensitivity for class Y
FPR_Y <- FP_Y / (FP_Y + TN_Y)  # Fall-out for class Y
Precision_Y <- TP_Y / (TP_Y + FP_Y)  # Precision for class Y
Recall_Y <- TPR_Y  # Recall is the same as TPR in this context
F_measure_Y <- 2 * (Precision_Y * Recall_Y) / (Precision_Y + Recall_Y)  # F-measure for class Y

# ROC and AUC
train_probabilities <- attr(predict(svm_model, x_train, probability = TRUE), "probabilities")[, "Y"]  # Get probabilities for class Y
roc_obj <- roc(y_train, train_probabilities)
roc_area <- auc(roc_obj)

# Calculate Matthews Correlation Coefficient (MCC)
MCC <- (TP_N * TN_N - FP_N * FN_N) / sqrt((TP_N + FP_N) * (TP_N + FN_N) * (TN_N + FP_N) * (TN_N + FN_N))

# Kappa statistic
kappa_stat <- conf_matrix$overall["Kappa"]

# Create performance measure table
performance_table <- data.frame(
  Class = c("N", "Y"),
  TPR = c(TPR_N, TPR_Y),
  FPR = c(FPR_N, FPR_Y),
  Precision = c(Precision_N, Precision_Y),
  Recall = c(Recall_N, Recall_Y),
  F_measure = c(F_measure_N, F_measure_Y),
  ROC = c(roc_area, roc_area),  # ROC area is the same for both classes
  MCC = c(MCC, MCC),  # MCC is the same for both classes
  Kappa = c(kappa_stat, kappa_stat)  # Kappa is the same for both classes
)

# Calculate weighted averages
weighted_avg <- data.frame(
  Class = "Weighted Average",
  TPR = sum(performance_table$TPR * table(y_train) / length(y_train)),
  FPR = sum(performance_table$FPR * table(y_train) / length(y_train)),
  Precision = sum(performance_table$Precision * table(y_train) / length(y_train)),
  Recall = sum(performance_table$Recall * table(y_train) / length(y_train)),
  F_measure = sum(performance_table$F_measure * table(y_train) / length(y_train)),
  ROC = roc_area,
  MCC = MCC,
  Kappa = kappa_stat
)

# Add weighted average row to the performance table
performance_table <- rbind(performance_table, weighted_avg)

# Print performance measure table
print(performance_table)

# Create ROC object using the true class labels and predicted probabilities
roc_obj <- roc(y_train, probabilities)
roc_area <- auc(roc_obj)
plot(roc_obj, 
     main = "ROC Curve for SVM", 
     col = "darkgreen", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,      # Shade the area under the curve
     auc.polygon.col = "lightgreen",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#4- XGBoost [B1,F3]
#------------------------------------
#------------------------------------

library(caret)
library(xgboost)
library(pROC)
library(dplyr)

# Assuming you have already trained your XGBoost model (xgb_model) and have your testing data (testing_data)
# Prepare the test data
x_test <- as.matrix(testing_data[, f3])  # Adjust this based on your test data's features
y_test <- as.numeric(factor(testing_data$Class, levels = c("N", "Y"))) - 1  # Convert target variable to numeric (0 and 1)

# Make predictions on the test data
predictions_test <- predict(xgb_model, x_test)  # Get predicted probabilities
predicted_classes_test <- ifelse(predictions_test > 0.5, 1, 0)  # Convert probabilities to class labels

# Create a confusion matrix
conf_matrix_test <- confusionMatrix(factor(predicted_classes_test), factor(y_test))
print(conf_matrix_test)

# Define a function to calculate performance measures
calculate_xgb_metrics <- function(actual, predicted) {
  # Ensure actual and predicted are numeric (0 or 1)
  actual <- as.numeric(actual)
  predicted <- as.numeric(predicted)
  
  # Create confusion matrix
  cm <- confusionMatrix(factor(predicted), factor(actual))
  
  # Extract counts for the confusion matrix
  TP_Y <- cm$table[2, 2]  # True Positives for Class Y
  FN_Y <- cm$table[2, 1]  # False Negatives for Class Y
  FP_Y <- cm$table[1, 2]  # False Positives for Class Y
  TN_Y <- cm$table[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm$table)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm$table)) / total
  
  # Kappa statistic
  kappa_stat <- cm$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(actual, predicted)  # Use numeric values (0 and 1)
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures for XGBoost:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  # Plot the ROC curve
  plot(roc_obj, 
       main = "ROC Curve for XGBoost", 
       col = "#009999", 
       lwd = 2, 
       print.auc = TRUE,        # Print AUC on the plot
       auc.polygon = TRUE,      # Shade the area under the curve
       auc.polygon.col = "#CCFFFF",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_xgb_metrics(y_test, predicted_classes_test)


#------------------------------------
#------------------------------------
#5- GBM [B1,F3]
#------------------------------------
#------------------------------------

library(gbm)
library(caret)
library(dplyr)
library(pROC)  

# Assuming `train_undersampled` contains your undersampled training data
# and `f3` contains the names of top features based on information gain
b1 <- train_undersampled[, c(f3, "Class")]  
b1$Class <- as.numeric(factor(b1$Class, levels = c("N", "Y"))) - 1  # N=0, Y=1

set.seed(123)

# Train the GBM model
gbm_model <- gbm(Class ~ ., data = b1,
                 distribution = "bernoulli",
                 n.trees = 100,      # Number of trees
                 interaction.depth = 3,  # Maximum depth of the trees
                 shrinkage = 0.01,   # Learning rate
                 bag.fraction = 0.5, # Proportion of trees to fit
                 cv.folds = 5,       # Cross-validation folds
                 n.cores = NULL)     # Use all available cores
summary(gbm_model)

# Make predictions on the training data
train_predictions <- predict(gbm_model, b1, n.trees = 100, type = "response")
predicted_classes_train <- ifelse(train_predictions > 0.5, 1, 0)

conf_matrix_train <- confusionMatrix(factor(predicted_classes_train), factor(b1$Class))
print(conf_matrix_train)

# Prepare the testing data
test_data <- testing_data[, c(f2, "Class")]
test_data$Class <- ifelse(test_data$Class == "Y", 1, 0)  # Convert to numeric binary format
predictions_prob <- predict(gbm_model, newdata = test_data, n.trees = 100, type = "response")
predictions <- ifelse(predictions_prob > 0.5, 1, 0)  # Thresholding at 0.5

# Generate the confusion matrix
conf_matrix <- confusionMatrix(as.factor(predictions), as.factor(test_data$Class))
print(conf_matrix$table)

# Function to calculate performance metrics for each class and weighted average
calculate_class_metrics <- function(actual, predicted, predicted_prob) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Calculate metrics for Class Y
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate (Sensitivity)
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N (actually TN from Class Y perspective)
  FN_N <- FP_Y  # False Negatives for Class N (actually FP from Class Y perspective)
  FP_N <- FN_Y  # False Positives for Class N (actually FN from Class Y perspective)
  TN_N <- TP_Y  # True Negatives for Class N (actually TP from Class Y perspective)
  
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Corrected MCC calculation
  mcc_numerator <- as.numeric((TP_Y * TN_Y) - (FP_Y * FN_Y))
  mcc_denominator <- sqrt(as.numeric((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y)))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # ROC area
  roc_obj <- roc(actual, predicted_prob)
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  # Plot ROC Curve
  plot(roc_obj, main = "ROC Curve for GBM Model", col = "blue", lwd = 2)
  abline(a = 0, b = 1, col = "red", lty = 2)  # Diagonal line for random guessing
}

calculate_class_metrics(test_data$Class, predictions, predictions_prob)

# Calculate the ROC curve for the GBM model
roc_curve <- roc(test_data$Class, predictions_prob)
plot(roc_curve, 
     main = "ROC Curve for GBM", 
     col = "orange", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,      # Shade the area under the curve
     auc.polygon.col = "lightyellow",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")




#------------------------------------
#------------------------------------
#6- KNN [B1,F3]
#------------------------------------
#------------------------------------

library(caret)
library(class)
library(dplyr)
library(pROC)  

# Prepare the dataset
b1 <- train_undersampled[, c(f3, "Class")]  # Select top features and Class column
b1$Class <- as.factor(b1$Class)  # Convert Class to factor

set.seed(123)

# Split into features and target variable
x_train <- scale(as.matrix(b1[, -ncol(b1)]))  # Normalize features
y_train <- b1$Class  # Target variable

# Train KNN model and make predictions
k <- 5
predicted_classes_train <- knn(x_train, x_train, y_train, k)

# Confusion matrix and performance metrics
conf_matrix_train <- confusionMatrix(predicted_classes_train, y_train)
print(conf_matrix_train)

# Function to calculate performance metrics
calculate_knn_metrics <- function(actual, predicted, predicted_prob) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Corrected MCC calculation
  mcc_numerator <- (TP_Y * TN_Y) - (FP_Y * FN_Y)
  mcc_denominator <- sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # ROC area
  roc_obj <- roc(actual, as.numeric(predicted))
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  

}


calculate_knn_metrics(y_train, predicted_classes_train, NULL)  # No predicted probabilities for KNN


cat("Overall Accuracy:", conf_matrix_train$overall['Accuracy'], "\n")
cat("Confusion Matrix:\n")
print(conf_matrix_train$table)


# Convert predicted classes to numeric for ROC calculations
predicted_numeric <- as.numeric(predicted_classes_train) - 1  # Assuming 'N' is 0 and 'Y' is 1

# Calculate the ROC curve
roc_obj <- roc(y_train, predicted_numeric)  # Using actual target variable and numeric predictions
plot(roc_obj, 
     main = "ROC Curve for KNN Model", 
     col = "purple", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lavender",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
# FOR BOOTSTRAPPING DATASET B2 
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------
#--------------------------------------------------------------------------


#--------
#--------
#BALANCING DATASET
#--------
#--------

# Find the indices of the majority (Class == "N") and minority class (Class == "Y")
zero_indices <- which(training_data$Class == "N")  # Updated to use train_data
one_indices <- which(training_data$Class == "Y")   # Updated to use train_data

# Count the number of indices for each class
class_counts <- c(length(zero_indices), length(one_indices))
class_names <- c("N", "Y")

# Create a data frame for visualization
data_vis <- data.frame(Class = class_names, Count = class_counts)

# Create a bar chart to visualize the class distribution
ggplot(data_vis, aes(x = Class, y = Count, fill = Class)) +
  geom_bar(stat = "identity") +
  labs(title = "Distribution of Indices by Class",
       x = "Class", y = "Count") +
  theme_minimal()

# Undersampling: randomly sample the majority class (zeros) to match the number of the minority class (ones)
z_indices <- sample(zero_indices, length(one_indices), replace = FALSE)

# Bootstrap resampling of both classes
bs_zero_indices <- sample(zero_indices, length(one_indices), replace = TRUE)
bs_one_indices <- sample(one_indices, length(one_indices), replace = TRUE)

# Bootstrap resampled training set
train_bs <- rbind(training_data[bs_zero_indices, ], training_data[bs_one_indices, ])
cat("Bootstrap resampled training set class distribution:\n")
table(train_bs$Class)

# Check the class distribution in the testing set
cat("Testing set class distribution:\n")
table(testing_data$Class)

dim(train_bs) #----------------------------BALANCED TRAINING SET-2 [B2]

#--------
#--------
#FEATURE SELECTION
#--------
#--------

#1 LASSO Regularization - [B2,F1]



library(glmnet)
library(caret)

# Prepare data
train_bs$Class <- as.factor(train_bs$Class)
x_train <- model.matrix(Class ~ . - 1, data = train_bs)  # No intercept in model matrix
y_train <- train_bs$Class

# Cross-validation for LASSO
set.seed(123)
cv_fit <- cv.glmnet(x_train, y_train, alpha = 1, family = "binomial", type.measure = "class")

# Plot and get best lambda
plot(cv_fit)
best_lambda <- cv_fit$lambda.min
cat("Best lambda:", best_lambda, "\n")

# Train final LASSO model
final_model <- glmnet(x_train, y_train, alpha = 1, lambda = best_lambda, family = "binomial")
coef_values <- as.matrix(coef(final_model))
coef_df <- data.frame(Feature = rownames(coef_values), Coefficient = coef_values[, 1])
important_features <- coef_df[coef_df$Coefficient != 0, ]
important_features <- important_features[important_features$Feature != "(Intercept)", ]

f1 <- important_features$Feature
cat("Highest importance features:\n")
print(important_features)
cat("Features selected for further use (f1):\n")
print(f1)


#+-+-+-+-+-+-+-+-+-+-+-
#+#+-+-+-+-+-+-+-+-+-+-+-

#2 - PCA (F2)

library(ggplot2)
library(dplyr)

# Prepare the data for PCA (removing the Class column)
pca_data <- train_bs[, -which(names(train_bs) == "Class")]
pca_data <- pca_data[, sapply(pca_data, var) != 0]
pca_data_scaled <- scale(pca_data)

# Perform PCA
pca_result <- prcomp(pca_data_scaled, center = TRUE, scale. = TRUE)
loading_scores_df <- data.frame(
  Feature = rownames(pca_result$rotation),
  Score = pca_result$rotation[, 1]
)

# Sort all features by absolute loading score in descending order and store in 'f2'
f2 <- loading_scores_df %>%
  arrange(desc(abs(Score))) %>%
  pull(Feature)

cat("Features from PCA stored in f2 (ordered by loading score):\n")
print(f2)

ggplot(loading_scores_df, aes(x = reorder(Feature, Score), y = Score)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Features Ordered by PCA Loading Score", x = "Features", y = "Loading Score") +
  theme_minimal()


#+-+-+-+-+-+-+-+-+-+-+-
#+-+-+-+-+-+-+-+-+-+-+-

#3 - INFORMATION GAIN (F3)

library(caret)

# Train a logistic regression model with caret
model <- train(Class ~ ., data = train_bs, method = "glm", family = "binomial")
importance <- varImp(model)
importance_df <- as.data.frame(importance$importance)
top_features <- rownames(importance_df)[order(-importance_df[, 1])] 

f3 <- top_features
cat("Top features based on information gain stored in f3:\n")
print(f3)




#+-+-+-+-+-+-+-+-+-+-+-
#+-+-+-+-+-+-+-+-+-+-+-

# ML ALGORITHM
#------------------------------------
#------------------------------------

#FOR B2,F1- BOOTSTRAPPING AND LASSO

#------------------------------------
#------------------------------------
#1- RANDOM FOREST [B2,F1]
#------------------------------------
#------------------------------------

library(randomForest)
library(caret)
library(dplyr)
library(pROC)  

# Assuming you have already created f1 and train_bs as per your previous code
# Prepare the training data for Random Forest
x_train_rf <- train_bs[, f1, drop = FALSE]  # Select only the features from LASSO
y_train_rf <- train_bs$Class  # Class variable
train_data_rf <- data.frame(Class = y_train_rf, x_train_rf)

# Train the Random Forest model
set.seed(123)
rf_model <- randomForest(Class ~ ., data = train_data_rf, importance = TRUE)
print(rf_model)

# Predict on the training data and get predicted probabilities
rf_predictions <- predict(rf_model, newdata = x_train_rf, type = "response")
rf_probabilities <- predict(rf_model, newdata = x_train_rf, type = "prob")[,2]  # Get probabilities for the positive class (Y)

# Confusion Matrix
confusion_matrix_rf <- confusionMatrix(as.factor(rf_predictions), as.factor(y_train_rf))
print(confusion_matrix_rf)

# Function to calculate performance metrics
calculate_rf_metrics <- function(actual, predicted, predicted_prob) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Corrected MCC calculation
  mcc_numerator <- (TP_Y * TN_Y) - (FP_Y * FN_Y)
  mcc_denominator <- sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted_prob))  # Use predicted probabilities for ROC
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
}

calculate_rf_metrics(y_train_rf, rf_predictions, rf_probabilities)

# Plot ROC Curve
roc_obj <- roc(as.numeric(y_train_rf) - 1, rf_probabilities)  # Use the probabilities for the positive class
plot(roc_obj, 
     main = "ROC Curve for Random Forest Model", 
     col = "red", 
     lwd = 2, 
     legacy.axes = TRUE,
     auc.polygon.col = "pink",  # Color of the shaded area,
     print.auc = TRUE, 
     print.thres = "best",
     threshold = "best")


#------------------------------------
#------------------------------------
#2- DECISION TREE [B2,F1]
#------------------------------------
#------------------------------------

library(rpart)
library(rpart.plot)
library(caret)
library(pROC)  

# Prepare the training data for Decision Tree
x_train_dt <- train_bs[, f1, drop = FALSE]  # Select only the features from LASSO
y_train_dt <- train_bs$Class  # Class variable
train_data_dt <- data.frame(Class = y_train_dt, x_train_dt)

# Train the Decision Tree model
set.seed(123)
dt_model <- rpart(Class ~ ., data = train_data_dt, method = "class")
rpart.plot(dt_model)

# Predict on the training data
dt_predictions <- predict(dt_model, newdata = x_train_dt, type = "class")
dt_probabilities <- predict(dt_model, newdata = x_train_dt, type = "prob")[, 2]  # Get probabilities for the positive class (Y)

# Confusion Matrix
confusion_matrix_dt <- confusionMatrix(as.factor(dt_predictions), as.factor(y_train_dt))
print(confusion_matrix_dt)

# Function to calculate performance metrics
calculate_dt_metrics <- function(actual, predicted, predicted_prob) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Corrected MCC calculation
  mcc_numerator <- (TP_Y * TN_Y) - (FP_Y * FN_Y)
  mcc_denominator <- sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted_prob))  # Use predicted probabilities for ROC
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )

  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
}
# Create a confusion matrix
confusion_matrix_dt <- confusionMatrix(as.factor(dt_predictions), as.factor(y_train_dt))
print(confusion_matrix_dt)


calculate_dt_metrics(y_train_dt, dt_predictions, dt_probabilities)

# Create ROC object using the true class labels and predicted probabilities
roc_obj <- roc(as.numeric(y_train_dt) - 1, dt_probabilities)  # Convert factors to numeric
plot(roc_obj, 
     main = "ROC Curve for Decision Tree", 
     col = "blue", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,      # Shade the area under the curve
     auc.polygon.col = "lightblue",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#------------------------------------
#------------------------------------
#3- SVM [B2,F1]
#------------------------------------
#------------------------------------


library(e1071)       
library(caret)       
library(pROC)        
library(dplyr)       

# Assuming you have already created f1 and train_bs as per your previous code
# Prepare the training data for SVM
x_train_svm <- train_bs[, f1, drop = FALSE]  # Select only the features from LASSO
y_train_svm <- train_bs$Class  # Class variable
train_data_svm <- data.frame(Class = y_train_svm, x_train_svm)

# Train the SVM model
set.seed(123)
svm_model <- svm(Class ~ ., data = train_data_svm, kernel = "linear", probability = TRUE)

# Predict on the training data
svm_predictions <- predict(svm_model, newdata = x_train_svm)
svm_probabilities <- attr(predict(svm_model, newdata = x_train_svm, probability = TRUE), "probabilities")  # Get probabilities

# Create a confusion matrix
confusion_matrix_svm <- confusionMatrix(as.factor(svm_predictions), as.factor(y_train_svm))
print(confusion_matrix_svm)

# Function to calculate performance metrics
calculate_svm_metrics <- function(actual, predicted, predicted_prob) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Corrected MCC calculation
  mcc_numerator <- (TP_Y * TN_Y) - (FP_Y * FN_Y)
  mcc_denominator <- sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # ROC area
  roc_obj <- roc(as.numeric(as.factor(actual)) - 1, predicted_prob[, "Y"])  # Convert factors to numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
}

# Call the function with training data
calculate_svm_metrics(y_train_svm, svm_predictions, svm_probabilities)

# Assuming 'svm_probabilities' contains the predicted probabilities for class Y
roc_obj <- roc(as.numeric(as.factor(y_train_svm)) - 1, svm_probabilities[, "Y"])  # Convert factors to numeric
plot(roc_obj, 
     main = "ROC Curve for SVM Model", 
     col = "darkgreen", 
     lwd = 2, 
     print.auc = TRUE,        # Print AUC on the plot
     auc.polygon = TRUE,      # Shade the area under the curve
     auc.polygon.col = "lightgreen",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#4- XGBOOST [B2,F1]
#------------------------------------
#------------------------------------

library(xgboost)
library(caret)
library(pROC)
library(dplyr)

# Prepare the data for XGBoost
x_train_xgb <- as.matrix(train_bs[, f1])  # Convert to matrix for XGBoost
y_train_xgb <- as.numeric(train_bs$Class) - 1  # Convert factor to numeric (0 for "N", 1 for "Y")

# Train the XGBoost model
set.seed(123)
xgb_model <- xgboost(data = x_train_xgb, label = y_train_xgb, 
                     nrounds = 100, 
                     objective = "binary:logistic", 
                     eval_metric = "logloss", 
                     verbose = 0)

# Predict on the training data
xgb_predictions <- predict(xgb_model, newdata = x_train_xgb)
xgb_pred_classes <- ifelse(xgb_predictions > 0.5, "Y", "N")  # Convert probabilities to class labels

# Create a confusion matrix for the training set
confusion_matrix_xgb <- confusionMatrix(as.factor(xgb_pred_classes), as.factor(train_bs$Class))
print(confusion_matrix_xgb)

# Extract counts for the confusion matrix
cm_xgb <- confusion_matrix_xgb$table
TP_Y <- cm_xgb[2, 2]  # True Positives for Class Y
FN_Y <- cm_xgb[2, 1]  # False Negatives for Class Y
FP_Y <- cm_xgb[1, 2]  # False Positives for Class Y
TN_Y <- cm_xgb[1, 1]  # True Negatives for Class Y

# Calculate metrics for Class Y
tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
precision_Y <- TP_Y / (TP_Y + FP_Y)
recall_Y <- tpr_Y
f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)

# Calculate metrics for Class N
TP_N <- TN_Y  # True Positives for Class N
FN_N <- FP_Y  # False Negatives for Class N
FP_N <- FN_Y  # False Positives for Class N
TN_N <- TP_Y  # True Negatives for Class N

# Calculate metrics for Class N
tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
precision_N <- TP_N / (TP_N + FP_N)
recall_N <- tpr_N
f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)

# Calculate weighted average metrics
total_xgb <- sum(cm_xgb)
wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total_xgb
wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total_xgb
wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total_xgb
wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total_xgb
wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total_xgb

# Overall accuracy
accuracy_xgb <- sum(diag(cm_xgb)) / total_xgb

# Corrected MCC calculation
mcc_numerator_xgb <- (TP_Y * TN_Y) - (FP_Y * FN_Y)
mcc_denominator_xgb <- sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y))
mcc_xgb <- ifelse(mcc_denominator_xgb == 0, NA, mcc_numerator_xgb / mcc_denominator_xgb)

# Kappa statistic
kappa_stat_xgb <- confusion_matrix_xgb$overall["Kappa"]

# ROC area
roc_obj_xgb <- roc(as.numeric(train_bs$Class) - 1, xgb_predictions)  # Convert factors to numeric
roc_area_xgb <- auc(roc_obj_xgb)

# Create performance measure table
metrics_xgb <- data.frame(
  Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
  Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area_xgb, mcc_xgb, kappa_stat_xgb),
  Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area_xgb, mcc_xgb, kappa_stat_xgb),
  Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area_xgb, mcc_xgb, kappa_stat_xgb)
)

cat("Performance Measures for XGBoost Model:\n")
print(metrics_xgb)
cat("Accuracy:", accuracy_xgb, "\n")

# Calculate ROC for XGBoost
roc_obj_xgb <- roc(as.numeric(train_bs$Class) - 1, xgb_predictions)  # Convert factors to numeric
plot(roc_obj_xgb, 
     main = "ROC Curve for XGBoost Model", 
     col = "#009999", 
     lwd = 2, 
     print.auc = TRUE,         # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "#CCFFFF",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")



#------------------------------------
#------------------------------------
#5- GBM [B2,F1]
#------------------------------------
#------------------------------------

library(caret)
library(pROC)

# Prepare the data for GBM
x_train_gbm <- train_bs[, f1]  # Select important features for training
y_train_gbm <- as.numeric(as.factor(train_bs$Class)) - 1  # Convert factor to numeric (0 for "N", 1 for "Y")
gbm_data <- data.frame(y = y_train_gbm, x_train_gbm)

# Train the GBM model
set.seed(123)
gbm_model <- gbm(formula = y ~ ., 
                 distribution = "bernoulli", 
                 data = gbm_data, 
                 n.trees = 100, 
                 interaction.depth = 3, 
                 shrinkage = 0.01, 
                 verbose = FALSE)

# Predict on the training data
gbm_predictions <- predict(gbm_model, newdata = x_train_gbm, n.trees = 100, type = "response")
gbm_pred_classes <- ifelse(gbm_predictions > 0.5, "Y", "N")  # Convert probabilities to class labels

# Ensure that actual classes are factors
actual_classes <- factor(train_bs$Class, levels = c("N", "Y"))

# Create a confusion matrix
confusion_matrix_gbm <- confusionMatrix(factor(gbm_pred_classes, levels = c("N", "Y")), actual_classes)
print(confusion_matrix_gbm)

# Function to calculate performance metrics
calculate_gbm_metrics <- function(actual, predicted, predicted_prob) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(factor(predicted, levels = c("N", "Y")), factor(actual, levels = c("N", "Y")))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Corrected MCC calculation
  mcc_numerator <- (TP_Y * TN_Y) - (FP_Y * FN_Y)
  mcc_denominator <- sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_Y))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # ROC area
  roc_obj <- roc(as.numeric(actual) - 1, predicted_prob)  # Convert factors to numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
}

calculate_gbm_metrics(actual_classes, gbm_pred_classes, gbm_predictions)

# Calculate ROC for GBM
roc_obj_gbm <- roc(as.numeric(train_bs$Class) - 1, gbm_predictions)  # Convert factors to numeric
plot(roc_obj_gbm, 
     main = "ROC Curve for GBM Model", 
     col = "orange", 
     lwd = 2, 
     print.auc = TRUE,         # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightyellow",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#5- KNN [B2,F1]
#------------------------------------
#------------------------------------

library(class)  
library(caret)  
library(pROC)   
library(dplyr)  

# Prepare the data for KNN
x_train_knn <- train_bs[, f1]  # Select important features for training
y_train_knn <- train_bs$Class  # Target variable
y_train_knn <- as.factor(y_train_knn)

# Set a value for k (number of neighbors)
k_value <- 5   
knn_predictions <- knn(train = x_train_knn, test = x_train_knn, cl = y_train_knn, k = k_value)

# Create a confusion matrix
confusion_matrix_knn <- confusionMatrix(as.factor(knn_predictions), y_train_knn)
print(confusion_matrix_knn)

# Function to calculate performance metrics
calculate_knn_metrics <- function(actual, predicted, x_train, y_train, k_value) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Manually calculating the probability of Class Y
  # Assuming equal weights for all neighbors, we'll count the number of neighbors that predicted Class Y
  predictions_matrix <- as.data.frame(matrix(NA, nrow = nrow(x_train), ncol = 1))
  for (i in 1:nrow(x_train)) {
    neighbors <- knn(train = x_train, test = x_train[i, ], cl = y_train, k = k_value, prob = TRUE)
    predictions_matrix[i, 1] <- mean(neighbors == "Y")  # Probability of predicting "Y"
  }
  prob_Y <- predictions_matrix[, 1]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, prob_Y)  # Convert factors to numeric
  roc_area <- auc(roc_obj)
  
  # Corrected MCC calculation
  mcc_numerator <- (TP_Y * TN_Y) - (FP_Y * FN_Y)
  mcc_denominator <- sqrt((TP_Y + FP_Y) * (TP_Y + FN_Y) * (TN_Y + FP_Y) * (TN_Y + FN_N))
  mcc <- ifelse(mcc_denominator == 0, NA, mcc_numerator / mcc_denominator)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, mcc, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, mcc, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, mcc, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
}

# Call the function with training data
calculate_knn_metrics(y_train_knn, knn_predictions, x_train_knn, y_train_knn, k_value)
plot(roc_obj, 
     main = "ROC Curve for KNN Model", 
     col = "purple", 
     lwd = 2, 
     print.auc = TRUE,         # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lavender",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------

#FOR B2,F2- BOOTSTRAPPING AND PCA

#------------------------------------
#------------------------------------
#1- RANDOM FOREST [B2,F2]
#------------------------------------
#------------------------------------

library(randomForest) 
library(caret)         
library(pROC)          
library(dplyr)         

# Prepare the data for Random Forest
x_train_rf <- train_bs[, f2]   # Select important features for training
y_train_rf <- train_bs$Class    # Target variable
y_train_rf <- as.factor(y_train_rf)

# Train the Random Forest model
set.seed(123)
rf_model <- randomForest(x = x_train_rf, y = y_train_rf, ntree = 100)

# Make predictions on the training data
rf_predictions <- predict(rf_model, newdata = x_train_rf)

# Create a confusion matrix
confusion_matrix_rf <- confusionMatrix(as.factor(rf_predictions), as.factor(y_train_rf))
print(confusion_matrix_rf)

# Function to calculate performance metrics
calculate_rf_metrics <- function(actual, predicted) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)  # Convert factors to numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
}

# Call the function with training data
calculate_rf_metrics(y_train_rf, rf_predictions)
plot(roc_obj, 
     main = "ROC Curve for Random Forest Model", 
     col = "red", 
     lwd = 2, 
     print.auc = TRUE,         # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightpink",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#2- DECISION TREE [B2,F2]
#------------------------------------
#------------------------------------

library(rpart)        
library(caret)       
library(pROC)         
library(dplyr)       

# Prepare the data for Decision Tree
x_train_dt <- train_bs[, f2]   # Select important features for training
y_train_dt <- train_bs$Class    # Target variable
train_dt <- data.frame(Class = y_train_dt, x_train_dt)

# Train the Decision Tree model
set.seed(123)
dt_model <- rpart(Class ~ ., data = train_dt, method = "class")

# Make predictions on the training data
dt_predictions <- predict(dt_model, newdata = x_train_dt, type = "class")

# Create a confusion matrix
confusion_matrix_dt <- confusionMatrix(as.factor(dt_predictions), as.factor(y_train_dt))
print(confusion_matrix_dt)

# Function to calculate performance metrics
calculate_dt_metrics <- function(actual, predicted) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)  # Convert factors to numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )

  cat("Performance Measures:\n")
  cat("Accuracy:", accuracy, "\n")
}

# Call the function with training data
calculate_dt_metrics(y_train_dt, dt_predictions)
plot(roc_obj, 
     main = "ROC Curve for Decision Tree Model", 
     col = "blue", 
     lwd = 2, 
     print.auc = TRUE,         # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightblue",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#3- SVM [B2,F2]
#------------------------------------
#------------------------------------

library(e1071)      
library(caret)      
library(pROC)       
library(dplyr)     

# Prepare the data for SVM
x_train_svm <- train_bs[, f2]   # Select important features for training
y_train_svm <- train_bs$Class    # Target variable

# Combine features and target variable into one data frame
train_svm <- data.frame(Class = y_train_svm, x_train_svm)

# Train the SVM model
set.seed(123)
svm_model <- svm(Class ~ ., data = train_svm, kernel = "linear")  # Linear kernel

# Make predictions on the training data
svm_predictions <- predict(svm_model, newdata = x_train_svm)
svm_probabilities <- predict(svm_model, newdata = x_train_svm, decision.values = TRUE)

# Create a confusion matrix
confusion_matrix_svm <- confusionMatrix(as.factor(svm_predictions), as.factor(y_train_svm))
print(confusion_matrix_svm)

# Function to calculate performance metrics
calculate_svm_metrics <- function(actual, predicted, probabilities) {
  # Create confusion matrix
  conf_matrix <- confusionMatrix(as.factor(predicted), as.factor(actual))
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)  # Use actual and predicted as numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )

  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
}

# Call the function with training data and predicted probabilities
calculate_svm_metrics(y_train_svm, svm_predictions, svm_probabilities)
plot(roc_obj, 
     main = "ROC Curve for Decision Tree Model", 
     col = "darkgreen", 
     lwd = 2, 
     print.auc = TRUE,         # Print AUC on the plot
     auc.polygon = TRUE,       # Shade the area under the curve
     auc.polygon.col = "lightgreen",  # Color of the shaded area
     xlab = "False Positive Rate (1 - Specificity)", 
     ylab = "True Positive Rate (Sensitivity)")


#------------------------------------
#------------------------------------
#4- XGBOOST [B2,F2]
#------------------------------------
#------------------------------------

library(xgboost)   
library(caret)     
library(pROC)      
library(dplyr)     

# Prepare the data for XGBoost
x_train_xgb <- train_bs[, f2]   # Select important features for training
y_train_xgb <- train_bs$Class    # Target variable

# Convert the target variable to a numeric format for XGBoost
y_train_xgb_numeric <- ifelse(y_train_xgb == "Y", 1, 0)  # Convert "Y" to 1 and "N" to 0

# Create a DMatrix object for XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(x_train_xgb), label = y_train_xgb_numeric)

# Set parameters for the XGBoost model
params <- list(
  objective = "binary:logistic",  # Binary classification
  eval_metric = "error",           # Evaluation metric
  max_depth = 3,                   # Maximum depth of the trees
  eta = 0.1,                       # Learning rate
  nthread = 2                      # Number of threads to use
)

# Train the XGBoost model
set.seed(123)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)

# Make predictions on the training data
xgb_predictions_numeric <- predict(xgb_model, dtrain)

# Convert probabilities to class labels
xgb_predictions <- ifelse(xgb_predictions_numeric > 0.5, "Y", "N")  # Convert probabilities to class labels

# Create a confusion matrix
confusion_matrix_xgb <- table(Predicted = xgb_predictions, Actual = y_train_xgb)
cat("Confusion Matrix:\n")
print(confusion_matrix_xgb)

# Function to calculate performance metrics
calculate_xgb_metrics <- function(actual, predicted, predicted_probs) {
  # Convert actual numeric labels back to factors for comparison
  actual_factor <- factor(ifelse(actual == 1, "Y", "N"), levels = c("N", "Y"))
  predicted_factor <- factor(predicted, levels = c("N", "Y"))
  
  # Create confusion matrix
  conf_matrix <- confusionMatrix(predicted_factor, actual_factor)
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, predicted_probs)  # Use actual as numeric and predicted_probs for ROC
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  # Plot ROC curve with specified parameters
  plot(roc_obj, 
       main = "ROC Curve for XGBoost Model", 
       col = "#009999", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "#CCFFFF",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}


calculate_xgb_metrics(y_train_xgb_numeric, xgb_predictions, xgb_predictions_numeric)


#------------------------------------
#------------------------------------
#5- GBM [B2,F2]
#------------------------------------
#------------------------------------

library(gbm)
library(caret)

# Prepare the data for GBM
x_train_gbm <- train_bs[, f2]  
y_train_gbm <- train_bs$Class  

# Convert the target variable to a numeric format for GBM
y_train_gbm_numeric <- ifelse(y_train_gbm == "Y", 1, 0)  

# Combine features and target variable into one data frame
train_gbm_data <- data.frame(y = y_train_gbm_numeric, x_train_gbm)

# Train the GBM model
set.seed(123)
gbm_model <- gbm(
  formula = y ~ .,  # Use 'y' directly from the combined data frame
  distribution = "bernoulli",  # Binary classification
  data = train_gbm_data,  # Data frame containing both features and target
  n.trees = 100,                # Number of trees
  interaction.depth = 3,        # Maximum depth of trees
  n.minobsinnode = 10,          # Minimum number of observations in the trees
  shrinkage = 0.1,              # Learning rate
  bag.fraction = 0.5,           # Fraction of data to be used for each tree
  train.fraction = 1,           # Use all training data
  n.cores = 1                   # Number of cores to use
)

# Make predictions on the training data
gbm_predictions_numeric <- predict(gbm_model, newdata = x_train_gbm, n.trees = gbm_model$n.trees, type = "response")
gbm_predictions <- ifelse(gbm_predictions_numeric > 0.5, "Y", "N")  

# Create a confusion matrix
confusion_matrix_gbm <- table(Predicted = gbm_predictions, Actual = y_train_gbm)
cat("Confusion Matrix:\n")
print(confusion_matrix_gbm)


# Function to calculate performance metrics
calculate_gbm_metrics <- function(actual, predicted, predicted_probs) {
  # Ensure predicted and actual are factors with the same levels
  actual_factor <- as.factor(actual)
  predicted_factor <- as.factor(predicted)
  
  # Set the same levels for both factors
  levels(predicted_factor) <- levels(actual_factor)
  
  # Create confusion matrix
  conf_matrix <- confusionMatrix(predicted_factor, actual_factor)
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, predicted_probs)  # Use actual as numeric and predicted_probs for ROC
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  # Plot ROC curve
  plot(roc_obj, 
       main = "ROC Curve for GBM Model", 
       col = "orange", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "lightyellow",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_gbm_metrics(y_train_gbm_numeric, gbm_predictions, gbm_predictions_numeric)


#------------------------------------
#------------------------------------
#6- KNN [B2,F2]
#------------------------------------
#------------------------------------

library(class)
library(caret)
library(pROC)

# Prepare the data for KNN
x_train_knn <- train_bs[, f2]  # Select important features from PCA
y_train_knn <- train_bs$Class   # Target variable
y_train_knn_numeric <- ifelse(y_train_knn == "Y", 1, 0)  # Convert to numeric for ROC

# Scale the feature data (KNN is sensitive to the scale of the data)
x_train_knn_scaled <- scale(x_train_knn)

# Make predictions using KNN
set.seed(123)
k <- 5  # Choose the number of neighbors
knn_predictions <- knn(train = x_train_knn_scaled, 
                       test = x_train_knn_scaled, 
                       cl = y_train_knn, 
                       k = k)

# Create a confusion matrix
confusion_matrix_knn <- table(Predicted = knn_predictions, Actual = y_train_knn)
cat("Confusion Matrix:\n")
print(confusion_matrix_knn)

# Function to calculate performance metrics
calculate_knn_metrics <- function(actual, predicted, predicted_probs) {
  # Ensure actual and predicted are factors
  actual_factor <- as.factor(actual)
  predicted_factor <- as.factor(predicted)
  
  # Create confusion matrix
  conf_matrix <- confusionMatrix(predicted_factor, actual_factor)
  cm <- conf_matrix$table
  
  # Extract counts for the confusion matrix
  TP_Y <- cm[2, 2]  # True Positives for Class Y
  FN_Y <- cm[2, 1]  # False Negatives for Class Y
  FP_Y <- cm[1, 2]  # False Positives for Class Y
  TN_Y <- cm[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm)) / total
  
  # Kappa statistic
  kappa_stat <- conf_matrix$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, predicted_probs)  # Use actual as numeric and predicted_probs for ROC
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )

  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  plot(roc_obj, 
       main = "ROC Curve for KNN Model", 
       col = "purple", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "lavender",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
  
}

calculate_knn_metrics(y_train_knn, knn_predictions, as.numeric(knn_predictions))



#------------------------------------
#------------------------------------

#FOR B2,F3- BOOTSTRAPPING AND INFORMATION GAIN

#------------------------------------
#------------------------------------
#1- RANDOM FOREST [B2,F3]
#------------------------------------
#------------------------------------


library(randomForest)
library(caret)
library(pROC)  
library(dplyr) 

# Prepare the data for Random Forest
x_train_rf <- train_bs[, f3]  
y_train_rf <- train_bs$Class   

# Train the Random Forest model
set.seed(123)
rf_model <- randomForest(x = x_train_rf, y = as.factor(y_train_rf), 
                         ntree = 100,       
                         mtry = round(sqrt(length(f3))),  
                         importance = TRUE)  

# Make predictions on the training data
rf_predictions <- predict(rf_model, newdata = x_train_rf)

# Create a confusion matrix
confusion_matrix_rf <- confusionMatrix(as.factor(rf_predictions), as.factor(y_train_rf))
cat("Confusion Matrix:\n")
print(confusion_matrix_rf)

# Function to calculate performance metrics
calculate_rf_metrics <- function(actual, predicted) {
  # Create confusion matrix
  cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
  
  # Extract counts for the confusion matrix
  TP_Y <- cm$table[2, 2]  # True Positives for Class Y
  FN_Y <- cm$table[2, 1]  # False Negatives for Class Y
  FP_Y <- cm$table[1, 2]  # False Positives for Class Y
  TN_Y <- cm$table[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm$table)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm$table)) / total
  
  # Kappa statistic
  kappa_stat <- cm$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)  # Use actual and predicted as numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  plot(roc_obj, 
       main = "ROC Curve for Random Forest Model", 
       col = "red", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "lightpink",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_rf_metrics(y_train_rf, rf_predictions)


#------------------------------------
#------------------------------------
#2- DECISION TREE [B2,F3]
#------------------------------------
#------------------------------------

library(rpart)
library(caret)
library(pROC)  
library(dplyr) 

# Prepare the data for Decision Tree
x_train_dt <- train_bs[, f3]  
y_train_dt <- train_bs$Class   

# Train the Decision Tree model
set.seed(123)
dt_model <- rpart(Class ~ ., data = train_bs[, c(f3, "Class")], method = "class")
dt_predictions <- predict(dt_model, newdata = x_train_dt, type = "class")

# Create a confusion matrix
confusion_matrix_dt <- confusionMatrix(as.factor(dt_predictions), as.factor(y_train_dt))
cat("Confusion Matrix:\n")
print(confusion_matrix_dt)

library(rpart.plot)
rpart.plot(dt_model, type = 3, extra = 101, fallen.leaves = TRUE)

# Function to calculate performance metrics
calculate_dt_metrics <- function(actual, predicted) {
  # Create confusion matrix
  cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
  
  # Extract counts for the confusion matrix
  TP_Y <- cm$table[2, 2]  # True Positives for Class Y
  FN_Y <- cm$table[2, 1]  # False Negatives for Class Y
  FP_Y <- cm$table[1, 2]  # False Positives for Class Y
  TN_Y <- cm$table[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm$table)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm$table)) / total
  
  # Kappa statistic
  kappa_stat <- cm$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)  # Use actual and predicted as numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  plot(roc_obj, 
       main = "ROC Curve for Decision Tree Model", 
       col = "blue", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "lightblue",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_dt_metrics(y_train_dt, dt_predictions)


#------------------------------------
#------------------------------------
#3- SVM [B2,F3]
#------------------------------------
#------------------------------------

library(e1071)
library(caret)
library(pROC)  
library(dplyr)

# Prepare the data for SVM
x_train_svm <- train_bs[, f3]  # Select important features from information gain
y_train_svm <- train_bs$Class   # Target variable
y_train_svm <- as.factor(y_train_svm)

# Train the SVM model
set.seed(123)
svm_model <- svm(x_train_svm, y_train_svm, probability = TRUE)
svm_predictions <- predict(svm_model, newdata = x_train_svm)

# Create a confusion matrix
confusion_matrix_svm <- confusionMatrix(as.factor(svm_predictions), as.factor(y_train_svm))
cat("Confusion Matrix:\n")
print(confusion_matrix_svm)

# Function to calculate performance metrics
calculate_svm_metrics <- function(actual, predicted) {
  # Create confusion matrix
  cm <- confusionMatrix(as.factor(predicted), as.factor(actual))
  
  # Extract counts for the confusion matrix
  TP_Y <- cm$table[2, 2]  # True Positives for Class Y
  FN_Y <- cm$table[2, 1]  # False Negatives for Class Y
  FP_Y <- cm$table[1, 2]  # False Positives for Class Y
  TN_Y <- cm$table[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm$table)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm$table)) / total
  
  # Kappa statistic
  kappa_stat <- cm$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)  # Use actual and predicted as numeric
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  plot(roc_obj, 
       main = "ROC Curve for SVM Model", 
       col = "darkgreen", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "lightgreen",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_svm_metrics(y_train_svm, svm_predictions)


#------------------------------------
#------------------------------------
#4- XGBOOST [B2,F3]
#------------------------------------
#------------------------------------

library(xgboost)
library(caret)
library(pROC) 
library(dplyr) 

# Prepare the data for XGBoost
x_train_xgb <- train_bs[, f3]  
y_train_xgb <- train_bs$Class  

# Convert the target variable to a numeric format for XGBoost
y_train_xgb_numeric <- ifelse(y_train_xgb == "Y", 1, 0)  

# Convert to DMatrix format for XGBoost
dtrain_xgb <- xgb.DMatrix(data = as.matrix(x_train_xgb), label = y_train_xgb_numeric)  

# Set parameters for the XGBoost model
params <- list(
  objective = "binary:logistic",  # Binary classification
  eval_metric = "logloss",         # Evaluation metric
  eta = 0.1,                       # Learning rate
  max_depth = 3                    # Maximum depth of trees
)

# Train the XGBoost model
set.seed(123)
xgb_model <- xgboost(data = dtrain_xgb, params = params, nrounds = 100)

# Make predictions on the training data
xgb_predictions_numeric <- predict(xgb_model, newdata = dtrain_xgb)
xgb_predictions <- ifelse(xgb_predictions_numeric > 0.5, "Y", "N")

# Create a confusion matrix
confusion_matrix_xgb <- confusionMatrix(as.factor(xgb_predictions), as.factor(y_train_xgb))
cat("Confusion Matrix:\n")
print(confusion_matrix_xgb)

calculate_xgb_metrics <- function(actual, predicted) {
  # Convert actual and predicted to factors with levels "N" and "Y"
  actual <- factor(actual, levels = c("N", "Y"))
  predicted <- factor(predicted, levels = c("N", "Y"))
  
  # Create confusion matrix
  cm <- confusionMatrix(predicted, actual)
  
  # Check if both classes are present
  if (nrow(cm$table) != 2) {
    stop("Both classes must be present in the predictions.")
  }
  
  # Extract counts for the confusion matrix
  TP_Y <- cm$table[2, 2]  # True Positives for Class Y
  FN_Y <- cm$table[2, 1]  # False Negatives for Class Y
  FP_Y <- cm$table[1, 2]  # False Positives for Class Y
  TN_Y <- cm$table[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N (handle missing classes)
  TP_N <- ifelse(!is.na(TN_Y), TN_Y, 0)  # True Positives for Class N
  FN_N <- ifelse(!is.na(FP_Y), FP_Y, 0)  # False Negatives for Class N
  FP_N <- ifelse(!is.na(FN_Y), FN_Y, 0)  # False Positives for Class N
  TN_N <- ifelse(!is.na(TP_Y), TP_Y, 0)  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm$table)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm$table)) / total
  
  # Kappa statistic
  kappa_stat <- cm$overall["Kappa"]
  
  # Calculate ROC area
  if (length(unique(actual)) > 1 && length(unique(predicted)) > 1) {
    roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)  # Use actual and predicted as numeric
    roc_area <- auc(roc_obj)
  } else {
    roc_area <- NA
  }
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  plot(roc_obj, 
       main = "ROC Curve for XGBoost Model", 
       col = "#009999", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "#CCFFFF",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_xgb_metrics(y_train_xgb, xgb_predictions)


#------------------------------------
#------------------------------------
#5- GBM [B2,F3]
#------------------------------------
#------------------------------------

library(gbm)
library(caret)
library(pROC)

# Prepare the data for GBM
x_train_gbm <- train_bs[, f3]  
y_train_gbm <- train_bs$Class    
y_train_gbm_numeric <- ifelse(y_train_gbm == "Y", 1, 0)  # Convert "Y" to 1 and "N" to 0

# Train the GBM model
set.seed(123)  # For reproducibility
gbm_model <- gbm(
  formula = y_train_gbm_numeric ~ .,  # Response variable must be numeric
  distribution = "bernoulli",          # For binary classification
  data = train_bs,                     # Training data
  n.trees = 100,                       # Number of trees
  interaction.depth = 3,               # Maximum depth of trees
  n.minobsinnode = 10,                 # Minimum number of observations in the tree nodes
  shrinkage = 0.01,                    # Learning rate
  bag.fraction = 0.5,                  # Fraction of data to be used for each tree
  train.fraction = 1,                  # Fraction of data used for training
  verbose = FALSE                      # Disable output during training
)

# Make predictions on the training data
gbm_probabilities <- predict(gbm_model, newdata = train_bs, n.trees = 100, type = "response")
gbm_predictions <- ifelse(gbm_probabilities > 0.5, "Y", "N")  # Convert probabilities to class labels

# Create a confusion matrix
confusion_matrix_gbm <- table(Predicted = gbm_predictions, Actual = y_train_gbm)
cat("Confusion Matrix:\n")
print(confusion_matrix_gbm)

# Define a function to calculate performance measures
calculate_gbm_metrics <- function(actual, predicted) {
  # Convert actual and predicted to factors
  actual <- factor(actual, levels = c("N", "Y"))
  predicted <- factor(predicted, levels = c("N", "Y"))
  
  # Create confusion matrix
  cm <- confusionMatrix(predicted, actual)
  
  # Extract counts for the confusion matrix
  TP_Y <- cm$table[2, 2]  # True Positives for Class Y
  FN_Y <- cm$table[2, 1]  # False Negatives for Class Y
  FP_Y <- cm$table[1, 2]  # False Positives for Class Y
  TN_Y <- cm$table[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm$table)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm$table)) / total
  
  # Kappa statistic
  kappa_stat <- cm$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  plot(roc_obj, 
       main = "ROC Curve for GBM Model", 
       col = "orange", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "lightyellow",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_gbm_metrics(y_train_gbm, gbm_predictions)



#------------------------------------
#------------------------------------
#6- KNN [B2,F3]
#------------------------------------
#------------------------------------

# Load necessary libraries
library(caret)
library(class)
library(dplyr)
library(pROC)

# Prepare the data for KNN
x_train_knn <- train_bs[, f3]  
y_train_knn <- train_bs$Class  

# Convert the target variable to a factor for KNN
y_train_knn <- as.factor(y_train_knn)
k <- 5  # You can experiment with different values for k
x_train_knn_scaled <- scale(x_train_knn)

# Make predictions using KNN on the training set
set.seed(123)  # For reproducibility
knn_predictions <- knn(train = x_train_knn_scaled, test = x_train_knn_scaled, cl = y_train_knn, k = k)

# Create a confusion matrix
confusion_matrix_knn <- table(Predicted = knn_predictions, Actual = y_train_knn)
cat("Confusion Matrix for KNN:\n")
print(confusion_matrix_knn)

# Define a function to calculate performance measures
calculate_knn_metrics <- function(actual, predicted) {
  # Convert actual and predicted to factors
  actual <- factor(actual, levels = c("N", "Y"))
  predicted <- factor(predicted, levels = c("N", "Y"))
  
  # Create confusion matrix
  cm <- confusionMatrix(predicted, actual)
  
  # Extract counts for the confusion matrix
  TP_Y <- cm$table[2, 2]  # True Positives for Class Y
  FN_Y <- cm$table[2, 1]  # False Negatives for Class Y
  FP_Y <- cm$table[1, 2]  # False Positives for Class Y
  TN_Y <- cm$table[1, 1]  # True Negatives for Class Y
  
  # Calculate metrics for Class Y
  tpr_Y <- TP_Y / (TP_Y + FN_Y)  # True Positive Rate
  fpr_Y <- FP_Y / (FP_Y + TN_Y)  # False Positive Rate
  precision_Y <- TP_Y / (TP_Y + FP_Y)
  recall_Y <- tpr_Y
  f1_Y <- (2 * precision_Y * recall_Y) / (precision_Y + recall_Y)
  
  # Calculate metrics for Class N
  TP_N <- TN_Y  # True Positives for Class N
  FN_N <- FP_Y  # False Negatives for Class N
  FP_N <- FN_Y  # False Positives for Class N
  TN_N <- TP_Y  # True Negatives for Class N
  
  # Calculate metrics for Class N
  tpr_N <- TP_N / (TP_N + FN_N)  # True Positive Rate for Class N
  fpr_N <- FP_N / (FP_N + TN_N)  # False Positive Rate for Class N
  precision_N <- TP_N / (TP_N + FP_N)
  recall_N <- tpr_N
  f1_N <- (2 * precision_N * recall_N) / (precision_N + recall_N)
  
  # Calculate weighted average metrics
  total <- sum(cm$table)
  wt_tpr <- (tpr_Y * (TP_Y + FN_Y) + tpr_N * (TP_N + FN_N)) / total
  wt_fpr <- (fpr_Y * (FP_Y + TN_Y) + fpr_N * (FP_N + TN_N)) / total
  wt_precision <- (precision_Y * (TP_Y + FP_Y) + precision_N * (TP_N + FP_N)) / total
  wt_recall <- (recall_Y * (TP_Y + FN_Y) + recall_N * (TP_N + FN_N)) / total
  wt_f1 <- (f1_Y * (TP_Y + FN_Y) + f1_N * (TP_N + FN_N)) / total
  
  # Overall accuracy
  accuracy <- sum(diag(cm$table)) / total
  
  # Kappa statistic
  kappa_stat <- cm$overall["Kappa"]
  
  # Calculate ROC area
  roc_obj <- roc(as.numeric(actual) - 1, as.numeric(predicted) - 1)
  roc_area <- auc(roc_obj)
  
  # Create performance measure table
  metrics <- data.frame(
    Metric = c("TPR", "FPR", "Precision", "Recall", "F-Measure", "ROC Area", "MCC", "Kappa"),
    Class_N = c(tpr_N, fpr_N, precision_N, recall_N, f1_N, roc_area, NA, kappa_stat),
    Class_Y = c(tpr_Y, fpr_Y, precision_Y, recall_Y, f1_Y, roc_area, NA, kappa_stat),
    Weighted_Avg = c(wt_tpr, wt_fpr, wt_precision, wt_recall, wt_f1, roc_area, NA, kappa_stat)
  )
  
  cat("Performance Measures for KNN:\n")
  print(metrics)
  cat("Accuracy:", accuracy, "\n")
  
  plot(roc_obj, 
       main = "ROC Curve for KNN Model", 
       col = "purple", 
       lwd = 2, 
       print.auc = TRUE,         # Print AUC on the plot
       auc.polygon = TRUE,       # Shade the area under the curve
       auc.polygon.col = "lavender",  # Color of the shaded area
       xlab = "False Positive Rate (1 - Specificity)", 
       ylab = "True Positive Rate (Sensitivity)")
}

calculate_knn_metrics(y_train_knn, knn_predictions)

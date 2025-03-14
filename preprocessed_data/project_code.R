#Loading Data
data <- read.csv("/Users/anoushkasingh/Documents/Sem 2/Thurs-Data Mining/Project/project_data.csv", header= TRUE)
dim(data)


# 1- MISSING VALUES

library(ggplot2)

# Check for missing values in each column
missing_values <- sapply(data, function(x) sum(is.na(x)))
total_rows <- nrow(data)
missing_percentage <- (missing_values / total_rows) * 100

# Create a data frame for plotting, only for columns with missing values
missing_values_df <- data.frame(
  Column = names(missing_percentage[missing_percentage > 0]),
  MissingPercentage = missing_percentage[missing_percentage > 0]
)

# Plot missing values percentage
ggplot(missing_values_df, aes(x = MissingPercentage, y = reorder(Column, MissingPercentage))) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(title = "Percentage of Missing Values by Column",
       x = "Missing Values (%)",
       y = "Column") +
  theme_minimal()

"Class" %in% colnames(data_final) #To check the if 'Class' column is present in the dataframe

#1.1 REMOVING COLUMNS

# Check for missing values in each column
# View columns with missing values
missing_values <- sapply(data, function(x) sum(is.na(x)))
total_rows <- nrow(data)

missing_values[missing_values > 0]
missing_percentage <- (missing_values / total_rows) * 100
missing_percentage[missing_percentage > threshold]

# Set your chosen threshold
threshold <- 25  

# Identify columns with missing values percentage above the threshold
columns_to_remove <- names(missing_percentage[missing_percentage > threshold])

# Remove the identified columns from the dataset
data_cleaned <- data[, !(names(data) %in% columns_to_remove)]
dim(data_cleaned)



#1.2 HANDLING THE REMAINING MISSING VALUES

# Check missing values in the remaining columns
remaining_missing_values <- sapply(data_cleaned, function(x) sum(is.na(x)))
remaining_missing_percentage <- (remaining_missing_values / nrow(data_cleaned)) * 100

# View columns with missing values
remaining_missing_values[remaining_missing_values > 0]


# Impute missing values for numerical columns with mean or median
for (col in names(data_cleaned)) {
  if (is.numeric(data_cleaned[[col]])) {
    # Mean imputation
    data_cleaned[[col]][is.na(data_cleaned[[col]])] <- mean(data_cleaned[[col]], na.rm = TRUE)
    # Alternatively, use median imputation
    # data_cleaned[[col]][is.na(data_cleaned[[col]])] <- median(data_cleaned[[col]], na.rm = TRUE)
  }
}
# Verify that there are no more missing values
sum(is.na(data_cleaned))

data_cleaned$Class <- as.factor(data_cleaned$Class)
class(data_cleaned$Class)



#2 ZERO VARIANCE

# Extract numeric columns from data_cleaned, excluding 'Class' column
# Identify columns with zero variance
# Extract names of columns with zero variance
# Remove columns with zero variance from numeric_columns
# Add the 'Class' column back to the cleaned data

numeric_columns <- data_cleaned[, sapply(data_cleaned, is.numeric) & !(names(data_cleaned) %in% "Class")]
zero_variance_columns <- sapply(numeric_columns, function(x) sd(x, na.rm = TRUE) == 0)
zero_variance_columns <- names(zero_variance_columns)[zero_variance_columns]
numeric_columns_cleaned <- numeric_columns[, !(colnames(numeric_columns) %in% zero_variance_columns)]
numeric_columns_cleaned <- cbind(numeric_columns_cleaned, data_cleaned[, "Class", drop = FALSE])

cat("Dimensions of cleaned data:", dim(numeric_columns_cleaned), "\n")
cat("Removed columns with zero variance:\n", zero_variance_columns, "\n")




#3 CORRELEATION ANALYSIS


library(caret)

# Extract numeric columns from data_cleaned, excluding 'Class' column
numeric_columns <- data_cleaned[, sapply(data_cleaned, is.numeric) & !(names(data_cleaned) %in% "Class")]
numeric_columns <- na.omit(numeric_columns) # Ensure there are no missing values in numeric columns

# Remove columns with zero variance
zero_variance_columns <- sapply(numeric_columns, function(x) sd(x, na.rm = TRUE) == 0)
zero_variance_columns <- names(zero_variance_columns)[zero_variance_columns]
numeric_columns <- numeric_columns[, !(colnames(numeric_columns) %in% zero_variance_columns)]

# Calculate the correlation matrix for the numeric columns
correlation_matrix <- cor(numeric_columns, use = "complete.obs")

correlation_threshold <- 0.8 # Set Correlation threshold

# Check if the correlation matrix still contains missing values
if (any(is.na(correlation_matrix))) {
  stop("The correlation matrix contains missing values. Please handle missing values before proceeding.")
}

# Use the findCorrelation function to identify highly correlated predictors
highly_correlated_indices <- findCorrelation(correlation_matrix, cutoff = correlation_threshold)

# Function to calculate average correlation for a given feature
average_correlation <- function(corr_matrix, feature_index) {
  feature_corr <- corr_matrix[, feature_index]
  mean(abs(feature_corr[!is.na(feature_corr) & !is.infinite(feature_corr)]))
}

# Initialize a vector to keep track of features to remove
features_to_remove <- c()

# Check if there are highly correlated features initially
  # Calculate average correlations for features to be removed
  # Identify the feature with the highest average correlation
  # Add feature to remove list
  # Remove feature from the dataset
  # Recalculate correlation matrix after removal
  # Update highly correlated indices

while (length(highly_correlated_indices) > 0) {
  avg_corr <- sapply(highly_correlated_indices, function(i) average_correlation(correlation_matrix, i))
  feature_to_remove <- highly_correlated_indices[which.max(avg_corr)]
  features_to_remove <- c(features_to_remove, feature_to_remove)
  numeric_columns <- numeric_columns[, -feature_to_remove]
  correlation_matrix <- cor(numeric_columns, use = "complete.obs")
  highly_correlated_indices <- findCorrelation(correlation_matrix, cutoff = correlation_threshold)
}

# Remove duplicates and finalize the dataset
features_to_remove <- unique(features_to_remove)
data_final <- cbind(numeric_columns, data_cleaned[, "Class", drop = FALSE])

cat("Dimensions of cleaned data:", dim(data_final), "\n")
cat("Removed features due to high correlation:\n", names(numeric_columns)[features_to_remove], "\n")



#4 BAR PLOT FOR CLASS

library(ggplot2)

class_counts <- as.data.frame(table(data_final$Class))

# Rename columns for clarity
colnames(class_counts) <- c("Class", "Count")

barplot(class_counts$Count,
        names.arg = class_counts$Class,
        main = "Count of Each Class",
        xlab = "Class",
        ylim = c(0,7000),
        ylab = "Count",
        col = "steelblue",
        border = "white")



#5 CHECKING FOR OUTLIERS

# Extract numeric columns from data_cleaned, excluding 'Class' column
numeric_columns <- data_cleaned[, sapply(data_cleaned, is.numeric)]

# Function to identify outliers using IQR
find_outliers_iqr <- function(x) {
  Q1 <- quantile(x, 0.25, na.rm = TRUE)  # 1st Quartile
  Q3 <- quantile(x, 0.75, na.rm = TRUE)  # 3rd Quartile
  IQR_value <- Q3 - Q1  # Interquartile Range
  
  
  lower_bound <- Q1 - 1.5 * IQR_value
  upper_bound <- Q3 + 1.5 * IQR_value
  
  return(x < lower_bound | x > upper_bound) # Return TRUE if outlier, FALSE otherwise
}

# Apply the function to each numeric column
outliers <- sapply(numeric_columns, find_outliers_iqr)
outlier_counts <- colSums(outliers, na.rm = TRUE)
outlier_counts

# Display the column with the maximum number of outliers
col_with_max_outliers <- names(which.max(outlier_counts))
max_outliers <- max(outlier_counts)

cat("The column with the maximum number of outliers is:", col_with_max_outliers, "with", max_outliers, "outliers.\n")

# Plot boxplots for all numeric columns to visually check for outliers
boxplot(numeric_columns, 
        main = "Boxplot of Numeric Columns", 
        col = "lightblue", 
        border = "black", 
        las = 2,  # Rotate labels for better readability
        outline = TRUE)  # Outliers will be displayed as points



#6 DOWNLOAD THE PRE-PROCESSED INTO .csv FILE

file_path <- "/Users/anoushkasingh/Documents/Sem 2/Thurs-Data Mining/Project/Intermediate Report/preprocessed_data.csv"
write.csv(data_final, file = file_path, row.names = FALSE)
cat("Data saved to:", file_path, "\n")


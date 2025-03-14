# Predictive Analytics for Asthma Related Emergency Visits
Predictive Analytics for Asthma-Related Emergency Visits: A Machine Learning Approach to Proactive Healthcare


---

## Overview  
This project focuses on predicting whether individuals with asthma will visit emergency rooms or urgent care centers within a year. Using the **2021 BRFSS Asthma Call-back Survey (ACBS)** dataset, I built a high-performance machine learning model to identify high-risk patients, enabling timely interventions and reducing healthcare costs.

---

## Key Objectives  
1. **Predict Asthma-Related Emergency Visits:** Classify individuals into "high-risk" (emergency visit) or "low-risk" (no visit) categories.
2. **Feature Selection:** Identify the most significant predictors of emergency visits using techniques like LASSO, PCA, and Information Gain.
3. **Model Optimization:** Build and evaluate multiple machine learning models to achieve the highest predictive accuracy.

---

## Tools & Technologies  
- **Programming Languages:** R, Python  
- **Libraries:** `caret`, `glmnet`, `xgboost`, `randomForest`, `ggplot2`, `scikit-learn`, `pandas`, `numpy`  
- **Techniques:**  
  - Data Preprocessing: Handling missing values, outlier detection, feature scaling  
  - Feature Selection: LASSO Regression, PCA, Information Gain  
  - Modeling: Random Forest, XGBoost, SVM, Gradient Boosting, Decision Trees, KNN  
  - Evaluation Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC  

---

## Process Overview  
1. **Data Preprocessing:**  
   - Handled missing values using mean imputation and removed columns with >25% missing data.  
   - Removed zero-variance features and highly correlated variables to reduce noise.  
   - Detected and analyzed outliers using the IQR method.  

2. **Balancing the Dataset:**  
   - Addressed class imbalance using **under-sampling** and **bootstrap resampling** techniques.  

3. **Feature Selection:**  
   - Applied **LASSO Regression**, **PCA**, and **Information Gain** to identify the most predictive features.  

4. **Model Building:**  
   - Trained and evaluated **6 classification algorithms**: Random Forest, XGBoost, SVM, Gradient Boosting, Decision Trees, and KNN.  
   - Used **stratified train-test splits** to ensure balanced evaluation.  

5. **Model Evaluation:**  
   - Evaluated models using **accuracy, precision, recall, F1-score, and ROC-AUC**.  
   - Selected the best-performing model: **XGBoost with Bootstrap Resampling and Information Gain**.  

6. **Results & Insights:**  
   - Achieved **93.47% accuracy** and **>92% TPR** for both classes.  
   - Identified key predictors of asthma-related emergency visits, such as **duration of asthma symptoms** and **hospital visits**.  

---

## Project Complexity  
This project involved building and evaluating **36 unique models** to ensure the best predictive performance. Here's how the complexity was structured:  

1. **2 Balancing Techniques:**  
   - **Under-sampling:** Randomly reduced the majority class to match the minority class.  
   - **Bootstrap Resampling:** Created balanced datasets by resampling with replacement.  

2. **3 Feature Selection Methods:**  
   - **LASSO Regression:** Identified key features by shrinking less important coefficients to zero.  
   - **PCA (Principal Component Analysis):** Reduced dimensionality by transforming features into principal components.  
   - **Information Gain:** Selected features based on their contribution to predicting the target variable.  

3. **6 Classification Algorithms:**  
   - **Random Forest**  
   - **XGBoost**  
   - **Support Vector Machine (SVM)**  
   - **Gradient Boosting Machine (GBM)**  
   - **Decision Trees**  
   - **K-Nearest Neighbors (KNN)**  

By combining **2 balancing techniques**, **3 feature selection methods**, and **6 classification algorithms**, I rigorously tested and compared **36 models** to identify the best-performing solution. This approach ensured robustness, reliability, and high predictive accuracy.  

---

## 📊 Results  
### Model Performance Summary  
| Model                          | Accuracy | Precision (Class Y) | Recall (Class Y) | F1-Score (Class Y) | ROC-AUC |
|--------------------------------|----------|---------------------|------------------|--------------------|---------|
| XGBoost (Bootstrap + InfoGain) | 93.47%   | 92.75%              | 94.30%           | 93.52%             | 0.98    |
| Random Forest                  | 89.12%   | 88.50%              | 89.80%           | 89.15%             | 0.94    |
| SVM                            | 86.59%   | 85.76%              | 88.94%           | 87.30%             | 0.92    |

---

## Visualizations  
- **ROC Curves:** Visualizing model performance across different thresholds.  
- **Feature Importance:** Highlighting the top predictors of emergency visits.  


---

## 🛠️ How to Run the Code  
1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/Asthma-Emergency-Visit-Prediction.git

2. Install dependencies:
pip install -r requirements.txt

4. Open the Jupyter/R notebooks in the notebooks/ folder to explore the analysis and modeling steps.

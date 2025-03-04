# Credit Card Fraud Detection

## Overview
Credit Card Fraud Detection project using a dataset from Kaggle, which includes 170884 credit card transactions with only 305 frauds, is highly unbalanced. To address the class imbalance, the project implements some models, achieving 0.85 PR_AUC.

## Dataset
The dataset used for this project consists of anonymized credit card transactions. Features include numerical values representing different aspects of transactions. The target variable indicates whether a transaction is fraudulent (`1`) or legitimate (`0`).

## Methodology
1. **Exploratory Data Analysis (EDA)**
   - Data cleaning and preprocessing
   - Feature distributions and correlations
   - Handling class imbalance using techniques like oversampling and undersampling

2. **Model Selection & Training**
   - Implemented machine learning models including Logistic Regression, Random Forest, and XGBoost
   - Feature engineering and selection
   - Model evaluation using precision, recall, F1-score, and PR-AUC

3. **Evaluation & Results**
   - Performance comparison across different models
   - Confusion matrix and classification report
   - PR curve analysis



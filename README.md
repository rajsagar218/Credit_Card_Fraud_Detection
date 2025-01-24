# Credit Card Fraud Detection

This project aims to develop a machine learning model to detect fraudulent credit card transactions in real-time, minimizing financial losses and improving customer experience. The model uses a publicly available dataset from Kaggle containing anonymized credit card transaction data. This README provides an overview of the steps taken in the project, including data preprocessing, model training, evaluation, and instructions to run the code and reproduce the results.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Methodology](#methodology)
- [Models Used](#models-used)
- [Evaluation Metrics](#evaluation-metrics)
- [Setup and Execution](#setup-and-execution)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Loading](#2-data-loading)
  - [3. Execute Code Cells](#3-execute-code-cells)
  - [4. Reproduce Results](#4-reproduce-results)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

---

## Project Overview

Credit card fraud is a significant problem for financial institutions and consumers, resulting in substantial financial losses and damage to customer trust. This project aims to develop a machine learning model to identify fraudulent transactions using the dataset provided. The key steps in the project are:

1. Data preprocessing: handling missing values, duplicate removal, and addressing class imbalance.
2. Training multiple machine learning models to detect fraud.
3. Evaluating the models using several performance metrics.
4. Selecting and saving the best performing model for future use.

---

## Data Source

The dataset used in this project is available on Kaggle. It contains anonymized credit card transaction data with the following features:

- Transaction amount
- Time
- Anonymized features derived from sensitive cardholder information
- Binary label indicating whether the transaction is fraudulent (1) or genuine (0)

The dataset is imbalanced, with fraudulent transactions being a minority class.

---

## Methodology

The methodology used for this project involves several key steps:

1. **Data Preprocessing**:
   - Handle missing values and duplicates.
   - Detect and handle outliers.
   - Split the dataset into training and testing sets (80% and 20%, respectively).

2. **Class Imbalance Handling**:
   - The Synthetic Minority Over-sampling Technique (SMOTE) is used to generate synthetic samples for the minority class (fraudulent transactions) to balance the dataset.

3. **Model Training and Evaluation**:
   - Two machine learning models were trained: **K-Nearest Neighbors (KNN)** and **Logistic Regression**.
   - Hyperparameter tuning was performed using **GridSearchCV** to optimize the models' performance.
   - Model evaluation is done using multiple metrics like accuracy, precision, recall, F1-score, and ROC AUC score.

4. **Model Deployment**:
   - The best performing model is selected and saved using **Pickle** for future use.

---

## Models Used

- **K-Nearest Neighbors (KNN)**: A simple and effective algorithm for classification based on the distance between data points.
- **Logistic Regression**: A linear classifier used to predict the probability of an outcome (fraud or non-fraud) based on input features.

---

## Evaluation Metrics

The following evaluation metrics are used to measure the performance of the models:

- **Accuracy**: The proportion of correct predictions (both fraudulent and non-fraudulent).
- **F1-Score**: A harmonic mean of precision and recall, useful for imbalanced datasets.
- **Recall**: The ability of the model to correctly identify fraudulent transactions.
- **ROC AUC Score**: Measures the performance of the model by plotting the true positive rate against the false positive rate.

---

## Setup and Execution

### 1. Environment Setup

- Create a new Google Colab notebook.
- Install the necessary libraries by running the following:
  
  ```python
  !pip install pandas scikit-learn imbalanced-learn matplotlib
  ```

### 2. Data Loading

- Upload the credit card transaction dataset (creditcard.csv) to your Google Drive.
- Mount your Google Drive in the notebook:

  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```

- Load the dataset using **Pandas**:

  ```python
  import pandas as pd
  data = pd.read_csv('/content/drive/MyDrive/creditcard.csv')
  ```

### 3. Execute Code Cells

- Follow the code cells sequentially in the notebook. These will perform the following operations:
  - Data preprocessing (handling missing values, duplicates, and outliers).
  - Split the dataset into training and testing sets.
  - Apply SMOTE to handle class imbalance.
  - Train both KNN and Logistic Regression models.
  - Hyperparameter tuning using GridSearchCV.
  - Model evaluation using various metrics.
  
### 4. Reproduce Results

- The final trained model and processed data are saved using **Pickle**.
- Load the saved model and data to reproduce the results:

  ```python
  import pickle
  with open('best_model.pkl', 'rb') as file:
      model = pickle.load(file)
  ```

---

## Results

The optimized KNN model achieved the following performance metrics on the test set:

- **Accuracy**: 0.9995
- **F1-Score**: 0.9995
- **Recall**: 1.0
- **ROC AUC Score**: 0.9995

These results show that the model is highly effective in detecting fraudulent transactions. The high recall score indicates that the model successfully identifies all fraudulent transactions.

---

## Conclusion

This project demonstrates the use of machine learning techniques, particularly K-Nearest Neighbors, to detect credit card fraud. By balancing the dataset using SMOTE and tuning the model's hyperparameters, we achieved exceptional performance metrics, making the model suitable for real-world fraud detection applications.

---

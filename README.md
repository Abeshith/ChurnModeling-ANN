# ChurnModelling ANN
# Understanding Artificial Neural Networks with TensorFlow/Keras

This repository contains Python code demonstrating the implementation of an Artificial Neural Network (ANN) using TensorFlow and Keras. The code is designed to predict customer churn using a dataset obtained from a bank.

## Overview

In this project, we build a simple ANN model using TensorFlow and Keras to predict customer churn based on various features provided in the dataset. The dataset contains information such as credit score, geography, gender, age, tenure, balance, number of products, etc.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Keras
- Pandas
- NumPy
- Matplotlib
- Scikit-learn

## Dataset

The dataset used in this project is named "Churn_Modelling.csv". It consists of 10,000 records with 14 columns including features like CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, and the target variable "Exited".

## Data Preprocessing

- Imported necessary libraries such as TensorFlow, Pandas, NumPy, Matplotlib, etc.
- Loaded the dataset using Pandas.
- Preprocessed the data by handling categorical variables like "Geography" and "Gender" using one-hot encoding.
- Split the dataset into features (X) and target variable (y).
- Split the data into training and testing sets using `train_test_split` from Scikit-learn.
- Scaled the features using `StandardScaler` from Scikit-learn.

## Building the Neural Network

- Imported required modules from TensorFlow and Keras for building the ANN model.
- Initialized the Sequential model.
- Added layers to the model:
  - Input layer using `tf.keras.layers.Input` with 11 units and ReLU activation function.
  - First hidden layer using `tf.keras.layers.Dense` with 7 units and ReLU activation function.
  - Second hidden layer using `tf.keras.layers.Dense` with 6 units and ReLU activation function.
  - Output layer using `tf.keras.layers.Dense` with 1 unit and sigmoid activation function.
- Compiled the model using the Adam optimizer with a custom learning rate and binary cross-entropy loss function.
- Utilized EarlyStopping callback to prevent overfitting and restore the best weights.
- Trained the model on the training data.

## Model Evaluation

- Plotted the model's training and validation accuracy and loss over epochs using Matplotlib.
- Evaluated the model's performance on the test data using accuracy score and confusion matrix.

## Results

- The model achieved an accuracy of approximately 85.9% on the test data.
- The confusion matrix revealed the model's performance in terms of true positives, true negatives, false positives, and false negatives.

## Conclusion

This project demonstrates the implementation of an Artificial Neural Network using TensorFlow and Keras for predicting customer churn. By following the steps outlined in this README, one can understand the process of building, training, and evaluating an ANN model using real-world data.

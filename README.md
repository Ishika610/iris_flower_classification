
# Iris Flower Classification

Welcome to the Iris Flower Classification project! This repository contains code and resources for classifying Iris flowers into different species using machine learning techniques.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
- [Results](#results)


## Overview

The goal of this project is to classify Iris flowers into one of three species (Setosa, Versicolour, or Virginica) based on four features: sepal length, sepal width, petal length, and petal width. This is a classic problem in the field of pattern recognition and machine learning.

## Dataset

The dataset for the code is already provided with name iris.csv

### Features

- `Sepal Length (cm)`
- `Sepal Width (cm)`
- `Petal Length (cm)`
- `Petal Width (cm)`
- `Species`: The class label (Setosa, Versicolour, Virginica)

## Installation

To get started, clone this repository to your local machine:

```
git clone https://github.com/your-username/iris-flower-classification.git
cd iris-flower-classification
```

Install the required dependencies using pip:

```
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: Prepare the data for modeling by loading the dataset and performing any necessary preprocessing steps.

```
python data_preprocessing.py
```

2. **Model Training**: Train various machine learning models to classify Iris flowers.

```
python train_model.py
```

3. **Prediction**: Use the trained model to classify new Iris flower samples.

```
python predict.py
```

## Models

The project explores multiple classification algorithms:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree Classifier
- Random Forest Classifier
- Gradient Boosting Classifier

## Results

The performance of each model is evaluated using metrics such as Accuracy, Precision, Recall, and F1 Score. Below are the results for each model:

| Model                    | Accuracy | Precision | Recall | F1 Score |
|--------------------------|----------|-----------|--------|----------|
| Logistic Regression      | 0.97     | 0.97      | 0.97   | 0.97     |
| K-Nearest Neighbors      | 0.96     | 0.96      | 0.96   | 0.96     |
| Support Vector Machine   | 0.98     | 0.98      | 0.98   | 0.98     |
| Decision Tree Classifier | 0.94     | 0.94      | 0.94   | 0.94     |
| Random Forest Classifier | 0.97     | 0.97      | 0.97   | 0.97     |
| Gradient Boosting Classifier | 0.97  | 0.97      | 0.97   | 0.97     |


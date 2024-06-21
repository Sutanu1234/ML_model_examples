# Fake News Detection

This project focuses on detecting fake news using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to classify news articles as real or fake based on their content.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Prediction](#prediction)
- [Requirements](#requirements)
- [Download CSV File](#download-csv-file)
- [Acknowledgements](#acknowledgements)

## Introduction
The spread of fake news has become a significant problem in today's digital age. This project aims to develop a predictive model that can distinguish between real and fake news articles. By leveraging techniques from NLP and machine learning, we can preprocess textual data and train a classifier to identify fake news with reasonable accuracy.

## Dataset
The dataset used in this project consists of news articles with labels indicating whether they are real or fake. The data includes the author's name and the title of the article. 

## Preprocessing
1. **Handling Missing Values**: All missing values in the dataset are replaced with empty strings.
2. **Merging Columns**: The `author` and `title` columns are merged to form a new `content` column.
3. **Text Cleaning and Stemming**: 
    - Non-alphabetic characters are removed.
    - The text is converted to lowercase.
    - Stopwords are removed.
    - Words are stemmed to their root form using PorterStemmer.

## Model Training
1. **Vectorization**: The textual data is converted into numerical data using the TF-IDF vectorizer.
2. **Splitting the Data**: The dataset is split into training and testing sets with a test size of 20%.
3. **Training the Model**: A Logistic Regression model is trained on the training data.

## Evaluation
The model is evaluated using accuracy scores on both the training and test datasets:
- **Training Data Accuracy**: Measures how well the model performs on the training data.
- **Test Data Accuracy**: Measures how well the model generalizes to unseen data.

## Prediction
The model can be used to predict whether a new article is real or fake. An example of making a prediction on a single test instance is provided.

## Requirements
- Python 3.x
- NumPy
- Pandas
- NLTK
- Scikit-learn

## Download CSV File
To run this project, you need to have the dataset. You can download the CSV file from the following link:
- [Download train.csv](https://www.kaggle.com/competitions/fake-news/data?select=train.csv)

After downloading, make sure to place the `train.csv` file in the same directory as your script.

## Acknowledgements
- The dataset used in this project is publicly available and has been used for educational purposes.
- The project structure and code are inspired by various open-source projects and tutorials on NLP and machine learning.

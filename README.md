# IMDB Movie Reviews Sentiment Analysis

## Project Overview

This project aims to perform sentiment analysis on the IMDB Movie Reviews dataset. The goal is to classify movie reviews as either positive or negative using Natural Language Processing (NLP) techniques and machine learning algorithms.

## Steps Involved

1. **Loading the Dataset**: 
   - The dataset is loaded using Pandas and initial exploratory data analysis is performed by displaying the first few rows, the shape, and information about the dataset.

2. **Data Preprocessing**:
   - **Replacing Sentiment Values**: The sentiment column is converted from categorical values ('positive' and 'negative') to numerical values (1 and 0).
   - **Removing HTML Tags**: HTML tags are removed from the reviews using regular expressions.
   - **Converting to Lowercase**: All text is converted to lowercase to ensure uniformity.
   - **Removing Special Characters**: Special characters are removed from the text, leaving only alphanumeric characters.
   - **Removing Stopwords**: Common English stopwords are removed using the NLTK library.
   - **Stemming Words**: Words are reduced to their root forms using the Porter Stemmer from NLTK.
   - **Joining Words**: The processed words are joined back into a single string for each review.

3. **Feature Extraction**:
   - **Count Vectorization**: The reviews are converted into a matrix of token counts using `CountVectorizer` with a maximum of 5000 features.

4. **Splitting the Dataset**:
   - The dataset is split into training and testing sets using an 80-20 split.

5. **Training the Classifiers**:
   - Three different Naive Bayes classifiers are initialized and trained on the training data:
     - Gaussian Naive Bayes
     - Multinomial Naive Bayes
     - Bernoulli Naive Bayes

6. **Predicting and Evaluating**:
   - The trained classifiers predict the sentiment of the test data.
   - The accuracy of each classifier is calculated and printed.

## Results

The accuracy scores of the classifiers are printed, providing insights into their performance:

- **Gaussian Naive Bayes Accuracy**
- **Multinomial Naive Bayes Accuracy**
- **Bernoulli Naive Bayes Accuracy**

## Libraries and Tools Used

- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **NLTK**: For natural language processing tasks.
- **Scikit-learn**: For machine learning algorithms and tools.

## Conclusion

This project demonstrates a complete workflow for sentiment analysis using NLP techniques and machine learning. The steps include data preprocessing, feature extraction, model training, and evaluation. The accuracy of different Naive Bayes classifiers is compared to determine the most effective model for this task.

## Download the Dataset

You can download the IMDB Movie Reviews dataset using the following link:

[Download IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

The dataset is provided by Stanford AI Lab and can be used for academic purposes.

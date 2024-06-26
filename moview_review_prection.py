import numpy as np
import pandas as pd

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')
print(df.head())  # Display the first 5 rows of the dataframe
print(df.shape)  # Display the shape of the dataframe
print(df.info())  # Display the info of the dataframe

# Replace sentiment values with numerical values
df['sentiment'].replace('positive', 1, inplace=True)
df['sentiment'].replace('negative', 0, inplace=True)

# Function to remove HTML tags
def clean_html(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

# Apply the function to remove HTML tags from the review column
df['review'] = df['review'].apply(clean_html)

# Function to convert text to lowercase
def convert_to_lower(text):
    return text.lower()

# Apply the function to convert reviews to lowercase
df['review'] = df['review'].apply(convert_to_lower)

# Function to remove special characters
def remove_special(text):
    x = ''
    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + ' '
    return x

# Apply the function to remove special characters from the reviews
df['review'] = df['review'].apply(remove_special)

import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Function to remove stopwords
def remove_stopwords(text):
    y = []
    for i in text.split():
        if i not in stopwords.words('english'):
            y.append(i)
    return y

# Apply the function to remove stopwords from the reviews
df['review'] = df['review'].apply(remove_stopwords)

from nltk.stem import PorterStemmer
ps = PorterStemmer()

# Function to stem words
def stem_words(text):
    y = []
    for i in text:
        y.append(ps.stem(i))
    return y   

# Apply the function to stem words in the reviews
df['review'] = df['review'].apply(stem_words)

# Function to join the words back into a single string
def join_back(list_input):
    return " ".join(list_input)

# Apply the function to join the stemmed words back into a single string
df['review'] = df['review'].apply(join_back)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)

# Convert the reviews to a matrix of token counts
X = cv.fit_transform(df['review']).toarray()
y = df['sentiment']

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Initialize the classifiers
clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()

# Train the classifiers
clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

# Predict the sentiment for the test set
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)

from sklearn.metrics import accuracy_score

# Calculate and print the accuracy scores
print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred1)}")
print(f"Multinomial Naive Bayes Accuracy: {accuracy_score(y_test, y_pred2)}")
print(f"Bernoulli Naive Bayes Accuracy: {accuracy_score(y_test, y_pred3)}")

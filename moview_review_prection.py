import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score

# Ensure stopwords are downloaded
nltk.download('stopwords')

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')
#print(df.head())  # Display the first 5 rows of the dataframe
#print(df.shape)  # Display the shape of the dataframe
#print(df.info())  # Display the info of the dataframe

# Replace sentiment values with numerical values
df['sentiment'].replace({'positive': 1, 'negative': 0}, inplace=True)

# Function to clean text
def clean_text(text):
    text = re.sub('<.*?>', '', text)  # Remove HTML tags
    text = text.lower()  # Convert to lowercase
    text = re.sub('[^a-z0-9]', ' ', text)  # Remove special characters
    return text

# Apply text cleaning
df['review'] = df['review'].apply(clean_text)

# Function to remove stopwords and stem words
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

# Apply preprocessing
df['review'] = df['review'].apply(preprocess_text)

# Convert the reviews to a matrix of token counts
cv = CountVectorizer(max_features=5000)
X = cv.fit_transform(df['review']).toarray()
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Initialize and train the classifiers
clf1 = GaussianNB()
clf2 = MultinomialNB()
clf3 = BernoulliNB()

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)

# Predict the sentiment for the test set
y_pred1 = clf1.predict(X_test)
y_pred2 = clf2.predict(X_test)
y_pred3 = clf3.predict(X_test)

# Calculate and print the accuracy scores
print(f"Gaussian Naive Bayes Accuracy: {accuracy_score(y_test, y_pred1)}")
print(f"Multinomial Naive Bayes Accuracy: {accuracy_score(y_test, y_pred2)}")
print(f"Bernoulli Naive Bayes Accuracy: {accuracy_score(y_test, y_pred3)}")


# Function to predict sentiment of a new review
def predict_sentiment(review):
    # Clean the review
    review = clean_text(review)
    # Preprocess the review
    review = preprocess_text(review)
    # Convert to token counts
    review_vector = cv.transform([review]).toarray()
    # Predict using Multinomial Naive Bayes (you can choose any of the trained models)
    prediction = clf2.predict(review_vector)
    # Return sentiment
    return 'positive' if prediction == 1 else 'negative'

# Take user input
user_review = input("Enter a movie review: ")

# Predict and print the sentiment
predicted_sentiment = predict_sentiment(user_review)
print(f"The predicted sentiment for the review is: {predicted_sentiment}")

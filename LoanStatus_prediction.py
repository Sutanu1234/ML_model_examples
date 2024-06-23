#importing dependencies
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

#data collection
loan_data = pd.read_csv('dataset.csv')

loan_data.head()

#number of rows and columns
loan_data.shape

loan_data.describe()

#number of missing values
loan_data.isnull().sum()

# Drop rows with missing values
loan_data = loan_data.dropna()

# Label encoding (replace categorical values with numerical values)
loan_data.replace({"Loan_Status": {'N': 0, 'Y': 1}}, inplace=True)

# Dependents column values
loan_data['Dependents'].value_counts()

# Replacing '3+' with 4
loan_data = loan_data.replace(to_replace='3+', value=4)

# Data visualization
# sns.countplot(x='Education', hue='Loan_Status', data=loan_data)

# Convert categorical columns to numerical values
loan_data.replace({
    'Married': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 1, 'Female': 0},
    'Self_Employed': {'No': 0, 'Yes': 1},
    'Property_Area': {'Rural': 0, 'Semiurban': 1, 'Urban': 2},
    'Education': {'Graduate': 1, 'Not Graduate': 0}
}, inplace=True)

# Separating features and target
x = loan_data.drop(columns=['Loan_ID', 'Loan_Status'], axis=1)
y = loan_data['Loan_Status']

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=2, stratify=y)

# Model training
classifier = svm.SVC(kernel='linear')
classifier.fit(x_train, y_train)

# Accuracy calculation
x_train_prediction = classifier.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
print("Training data accuracy:", training_data_accuracy)

x_test_prediction = classifier.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print("Test data accuracy:", test_data_accuracy)

# Making a prediction system
input_data = (1,1,0,0,0,7660,0,104,360,0,2)

# Convert input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the numpy array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = classifier.predict(input_data_reshaped)
print("Prediction:", prediction)

if prediction[0] == 0:
    print('Not approved for loan')
else:
    print('Approved for loan')

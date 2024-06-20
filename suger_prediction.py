import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Loading dataset
dataset = pd.read_csv('diabetes.csv')

# Displaying dataset information
print(dataset.head())
print(dataset.shape)
print(dataset.describe())
print(dataset['Outcome'].value_counts())
print(dataset.groupby('Outcome').mean())

# Separating data and labels
X = dataset.drop(columns='Outcome', axis=1)
Y = dataset['Outcome']
print(X)
print(Y)

# Standardizing data
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
print(standardized_data)

X = standardized_data
Y = dataset['Outcome']

# Splitting data into train and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
print(X.shape, X_train.shape, X_test.shape)

# Training the SVM classifier
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)

# Model evaluation
# Accuracy score
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Accuracy score of the training data: ', training_data_accuracy)

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score of the test data: ', test_data_accuracy)

# Predicting with new data
input_data = (2, 197, 70, 45, 543, 30.5, 0.158, 53)

# Converting the input data to a numpy array and creating a DataFrame
input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
input_data_df = pd.DataFrame(input_data_as_numpy_array, columns=dataset.columns[:-1])

# Standardizing the input data
std_data = scaler.transform(input_data_df)
print(std_data)

# Making a prediction
prediction = classifier.predict(std_data)
print(prediction)

if prediction[0] == 0:
    print('The person does not have diabetes')
else:
    print('The person has diabetes')

# Plotting

# 1. Distribution of each feature
plt.figure(figsize=(15, 10))
for i, column in enumerate(dataset.columns[:-1], 1):
    plt.subplot(3, 3, i)
    sns.histplot(dataset[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()

# 2. Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(dataset.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# 3. Pairplot (visualize relationships)
sns.pairplot(dataset, hue='Outcome')
plt.show()

# Visualizing SVM decision boundary (only possible with 2D data)
# For the sake of visualization, let's select only two features
X_vis = dataset[['Glucose', 'BMI']]
Y_vis = dataset['Outcome']

# Standardizing the data
scaler_vis = StandardScaler()
scaler_vis.fit(X_vis)
X_vis = scaler_vis.transform(X_vis)

# Splitting data into train and test sets
X_train_vis, X_test_vis, Y_train_vis, Y_test_vis = train_test_split(X_vis, Y_vis, test_size=0.2, stratify=Y_vis, random_state=42)

# Training the SVM classifier
classifier_vis = svm.SVC(kernel='linear')
classifier_vis.fit(X_train_vis, Y_train_vis)

# Plotting decision boundary
def plot_decision_boundary(X, y, model):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Glucose')
    plt.ylabel('BMI')
    plt.title('SVM Decision Boundary')
    plt.show()

plot_decision_boundary(X_train_vis, Y_train_vis, classifier_vis)

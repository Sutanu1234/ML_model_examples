#importing dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#deta collection and data processing
# loadin data set in pandas
sonar_data = pd.read_csv('sonar_data.csv', header=None) 
sonar_data.head()
# no of rows and colums
sonar_data.shape
sonar_data.describe()
sonar_data[60].value_counts()
sonar_data.groupby(60).mean()


# separeting data and labels
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
print(X)
print(Y)

#Traing and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=42)
print(X.shape, X_train.shape, X_test.shape)
print(X_train)
print(Y_train)

model = LogisticRegression()
#traning logistic regrassion model with traing data
model.fit(X_train, Y_train)

#accuracy of training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print('Acurecy on traing data:', training_data_accuracy)
#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)   
print('Accuracy score of test data : ', test_data_accuracy)

#making a predicting system
input_data = (0.0522,0.0437,0.0180,0.0292,0.0351,0.1171,0.1257,0.1178,0.1258,0.2529,0.2716,0.2374,0.1878,0.0983,0.0683,0.1503,0.1723,0.2339,0.1962,0.1395,0.3164,0.5888,0.7631,0.8473,0.9424,0.9986,0.9699,1.0000,0.8630,0.6979,0.7717,0.7305,0.5197,0.1786,0.1098,0.1446,0.1066,0.1440,0.1929,0.0325,0.1490,0.0328,0.0537,0.1309,0.0910,0.0757,0.1059,0.1005,0.0535,0.0235,0.0155,0.0160,0.0029,0.0051,0.0062,0.0089,0.0140,0.0138,0.0077,0.0031)

#cahnging input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]=='R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')
# Diabetes Prediction Project

## Project Objective

The objective of this project is to build a machine learning model that can predict whether a person has diabetes based on various medical attributes. The dataset used for this project is the Pima Indians Diabetes Database, which contains medical data of several patients, along with an outcome indicating whether or not the patient has diabetes.

## Steps Involved

### Loading and Exploring the Dataset

1. Load the dataset using `pandas`.
2. Display the first few rows of the dataset to understand its structure.
3. Display basic statistics of the dataset using the `describe` method.
4. Examine the distribution of the outcome variable (diabetes vs. no diabetes).

### Data Preprocessing

1. Separate the features (independent variables) from the target variable (dependent variable).
2. Standardize the feature data to ensure all features are on the same scale, which is crucial for many machine learning algorithms.
3. Split the data into training and testing sets to evaluate the model's performance on unseen data.

### Model Training

1. Use a Support Vector Machine (SVM) with a linear kernel to train the model on the training data.
2. Fit the model using the standardized training data.

### Model Evaluation

1. Make predictions on the training data and calculate the accuracy to see how well the model has learned from the data.
2. Make predictions on the test data and calculate the accuracy to evaluate how well the model generalizes to new, unseen data.

### Prediction on New Data

1. Standardize the new input data using the previously fitted scaler.
2. Make a prediction using the trained SVM model to determine if the new patient has diabetes.

### Visualization

1. Plot the distribution of each feature to understand their individual characteristics.
2. Plot a correlation heatmap to visualize relationships between different features.
3. Create a pair plot to visualize relationships between pairs of features, colored by the outcome.
4. Visualize the decision boundary of the SVM model (using only two features for simplicity).


## graphs:<br><br>

![image](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/3917042e-b6ec-4aee-bb2b-ef579f011827)

![image](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/2a612e54-e719-45dc-a1f2-1ebf5ba72e21)

![image](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/7a9ff4b7-5ccf-47fc-b2ed-f96595eab487)

![image](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/3922dfdc-d121-4cd8-a246-96a25a15b022)





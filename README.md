# Wine Quality Prediction

This project aims to predict the quality of red wine using a machine learning model. The dataset used is the Wine Quality Dataset, which includes various chemical properties of red wine and their respective quality ratings.

## Dataset

The dataset `winequality-red.csv` contains the following columns:
- fixed acidity
- volatile acidity
- citric acid
- residual sugar
- chlorides
- free sulfur dioxide
- total sulfur dioxide
- density
- pH
- sulphates
- alcohol
- quality

The target variable is `quality`, which is an integer value between 0 and 10.

## Project Steps

### 1. Loading the Dataset

The dataset is loaded into a Pandas DataFrame for easy manipulation and analysis.
```
wine_dataset = pd.read_csv('winequality-red.csv')
```

### 2. Data Exploration

- Display the shape and first 5 rows of the dataset.
- Check for missing values.
- Generate statistical measures of the dataset.

### 3. Data Visualization

Using Seaborn and Matplotlib to visualize the data:
- Count plot for each quality value.
- Bar plots to show the relationship between `volatile acidity`, `citric acid`, and wine `quality`.
- Heatmap to display the correlation between different features.

### Graphs

- **Number of Values for Each Quality**  
![quality_count](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/7ff64717-38c0-44df-829e-a6f9a8cb3f1f)

- **Volatile Acidity vs Quality**  
![volatile_acidity_vs_quality](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/11ce9342-88dd-48f0-b534-61c56859ab40)

- **Citric Acid vs Quality**  
![citric_acid_vs_quality](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/729e3f19-4606-4c26-bedb-c532de291838)

- **Correlation Heatmap**  
![correlation_heatmap](https://github.com/Sutanu1234/ML_model_examples/assets/123285380/543d1c19-0a95-40c1-a42b-691b88c4e49b)


### 4. Data Preprocessing

- Separate the features and the target variable.
- Convert the quality rating into a binary classification problem where quality >= 7 is considered good (1) and others are considered bad (0).
- Split the data into training and testing sets.

```
X = wine_dataset.drop('quality', axis=1)
Y = wine_dataset['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)
```

### 5. Model Training

A Random Forest Classifier is used to train the model.

```
model = RandomForestClassifier()
model.fit(X_train, Y_train)
```

### 6. Model Evaluation

Evaluate the model's accuracy on the test data.

```
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy : ', test_data_accuracy)
```

### 7. Predictive System

A predictive system is built to predict the quality of wine based on user input.

```
input_data = (7.5, 0.5, 0.36, 6.1, 0.071, 17.0, 102.0, 0.9978, 3.35, 0.8, 10.5)
prediction = model.predict(input_data_reshaped)
if prediction[0] == 1:
    print('Good Quality Wine')
else:
    print('Bad Quality Wine')
```

## Results

The accuracy of the model on the test data is printed, and a simple predictive system is implemented to classify the quality of wine based on input chemical properties.

## Conclusion

This project demonstrates the use of machine learning techniques to predict wine quality, providing insights into the importance of different chemical properties in determining the quality of wine. The Random Forest Classifier proves to be effective in this binary classification problem.

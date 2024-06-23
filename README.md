# Loan Eligibility Prediction

This project aims to predict loan eligibility based on various attributes of loan applicants using a Support Vector Machine (SVM) model. The dataset includes information about the applicants' demographics, education, employment, income, loan amount, credit history, and property area.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview
The primary goal of this project is to build a predictive model that can classify whether a loan will be approved or not based on historical data. The dataset used contains various features such as gender, marital status, number of dependents, education level, employment status, income details, loan amount, loan term, credit history, and property area.

## Dataset
The dataset used for this project is `dataset.csv`. It includes the following columns:
- `Loan_ID`: Unique Loan ID
- `Gender`: Gender of the applicant
- `Married`: Marital status
- `Dependents`: Number of dependents
- `Education`: Education level
- `Self_Employed`: Self-employed status
- `ApplicantIncome`: Applicant's income
- `CoapplicantIncome`: Co-applicant's income
- `LoanAmount`: Loan amount
- `Loan_Amount_Term`: Loan term in months
- `Credit_History`: Credit history (1: good, 0: bad)
- `Property_Area`: Property area (Rural, Semiurban, Urban)
- `Loan_Status`: Loan status (Y: approved, N: not approved)

## Installation
To run this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/loan-eligibility-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd loan-eligibility-prediction
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Ensure the `dataset.csv` file is in the project directory.
2. Run the script to train the model and make predictions:
    ```bash
    python loan_prediction.py
    ```

## Model
The project uses a Support Vector Machine (SVM) with a linear kernel for training the classification model. The data preprocessing steps include handling missing values, label encoding, and splitting the data into training and testing sets.

## Results
The model's performance is evaluated using accuracy metrics on both training and test data. Example accuracy results are:
- Training data accuracy: 81.11%
- Test data accuracy: 81.25%

Example prediction for an input data point:
```python
input_data = (1, 1, 0, 0, 0, 7660, 0, 104, 360, 0, 2)
```
The prediction result indicates whether the loan is approved or not.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the `MIT License`. See the `LICENSE` file for more details.

Save this content in a file named `README.md` in your project's root directory. This will provide an overview of the project and instructions for setting it up and using it.


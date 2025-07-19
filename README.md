# Loan Status Prediction Using Support Vector Machine (SVM)

This project uses a supervised machine learning model to predict whether a loan application will be approved or not, based on applicant details and financial information.

## Project Overview

- Dataset: train_u6lujuX_CVtuZ9i.csv
- Model: Support Vector Classifier (SVC with linear kernel)
- Libraries: NumPy, Pandas, Seaborn, Scikit-learn
- Environment: Google Colab / Jupyter Notebook

## Dataset Description

- Total records (after dropping nulls): 480
- Features:
  - Gender
  - Married
  - Dependents
  - Education
  - Self_Employed
  - ApplicantIncome
  - CoapplicantIncome
  - LoanAmount
  - Loan_Amount_Term
  - Credit_History
  - Property_Area
- Target:
  - Loan_Status (1 = Approved, 0 = Not Approved)

## Workflow Summary

1. **Data Collection and Cleaning**
   - Loaded dataset from CSV file
   - Dropped rows with missing values
   - Replaced categorical values (e.g., `Yes/No`, `3+`, etc.) with numerical codes

2. **Data Preprocessing**
   - Encoded binary and categorical variables using `replace()`
   - Mapped 'Loan_Status' target to 0 and 1
   - Converted all features to numeric format

3. **Feature and Target Selection**
   - Features `X`: All columns except `Loan_ID` and `Loan_Status`
   - Target `Y`: Loan_Status

4. **Train-Test Split**
   - Split data into 90% training and 10% testing sets using `train_test_split()`

5. **Model Training**
   - Trained an SVM model with a linear kernel on the training data

6. **Model Evaluation**
   - Accuracy on training data: 79.86%
   - Accuracy on test data: 83.33%

7. **Prediction Example**
   - Demonstrated predictions on custom user input
   - Output returns whether a loan is likely to be approved or not

## How to Run

1. Install required libraries
2.  Load the notebook `Loan status prediction.ipynb` in Colab or Jupyter.
3. Ensure the dataset file is named correctly and placed in the working directory.
4. Run the cells sequentially to train the model and test predictions.

## Notes
- All missing data was dropped to simplify the model training.
- Feature scaling was not required for the linear SVM in this case.
- Input data for prediction must follow the same format and ordering as the training features.

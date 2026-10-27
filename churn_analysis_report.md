# Predictive Churn Analysis Report

## 1. Introduction

Customer churn occurs when customers stop using a company's products or services. Predicting churn helps businesses identify customers who may leave and plan appropriate retention strategies.

The objective of this project is to build and evaluate a machine-learning classification pipeline that predicts customer churn probability.

## 2. Dataset Description

The dataset contains 800 customer records and 13 original columns. It includes customer age, region, tenure, subscription type, monthly charges, total charges, contract type, support tickets, late payments, payment method, automatic payment status, customer ID, and churn status.

The target variable is `Churn`, with values representing whether a customer churned.

Initial inspection found no missing values and no duplicate rows.

## 3. Data Preprocessing

The `CustomerID` column was excluded from the model because it is an identifier rather than a meaningful predictive feature.

Categorical features were transformed using One-Hot Encoding. Numerical features were scaled using MinMaxScaler.

A train-test split allocated 80% of the data (640 records) to training and 20% (160 records) to testing. Stratified splitting maintained the approximate class proportions.

The preprocessing steps were fitted within a Scikit-learn pipeline to avoid fitting transformations on the test set.

## 4. Model Development

Logistic Regression was selected as the baseline classification algorithm. The model was trained on the training data and evaluated on the held-out test data.

The model generated churn predictions and estimated churn probabilities for the test customers.

## 5. Model Evaluation

The model achieved the following results:

| Metric | Result |
|---|---:|
| Precision | 0.5763 |
| Recall | 0.5075 |
| F1-score | 0.5397 |
| ROC-AUC | 0.6808 |
| Accuracy | 64% |

### Interpretation

The precision score indicates that approximately 57.63% of customers predicted to churn were actual churners in the test set.

The recall score indicates that the model identified approximately 50.75% of the customers who actually churned.

The F1-score reflects the balance between precision and recall. The ROC-AUC score indicates moderate ability to distinguish churners from non-churners.

The results suggest that the baseline model has useful predictive signal but needs further improvement before practical deployment.

## 6. Visualizations and Prediction Output

The project generates:

- `roc_curve.png` — visualization of the model's ROC curve.
- `confusion_matrix.png` — comparison of actual and predicted classes.
- `churn_risk_predictions.csv` — test-set customer IDs, actual outcomes, predicted outcomes, churn probabilities, and illustrative risk categories.

The risk categories use probability bands of below 30% for low risk, 30% to below 60% for medium risk, and 60% or above for high risk. These thresholds are illustrative and have not been validated for business use.

## 7. Limitations and Future Improvements

The model's predictive performance is moderate. Future improvements could include comparing Logistic Regression with Decision Tree and Random Forest models, examining feature importance, testing classification thresholds, and validating performance with cross-validation.

The dataset contains 800 records, so results may not generalize to other customer populations. Additional real-world validation would be necessary before using the predictions to guide customer-retention decisions.

## 8. Conclusion

This project demonstrates a complete baseline churn-classification workflow, including data inspection, categorical encoding, numerical scaling, train-test splitting, model training, evaluation, visualization, and prediction export.

The Logistic Regression model achieved a ROC-AUC of 0.6808 and accuracy of 64% on the held-out test set. These results provide a baseline for further experimentation and improvement.


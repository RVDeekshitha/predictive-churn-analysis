# Predictive Churn Analysis

## Project Overview

This project uses machine learning to predict customer churn and estimate customer retention risk using customer subscription and service information.

## Objective

To build and evaluate a classification model that identifies customers who may leave a service.

## Dataset

- 800 customer records
- 13 original columns
- Target variable: `Churn`
- Missing values: None found during initial inspection
- Duplicate rows: None found during initial inspection

## Methodology

1. Loaded and inspected the customer churn dataset.
2. Excluded the customer ID from predictive features.
3. Encoded categorical variables using One-Hot Encoding.
4. Scaled numerical features using MinMaxScaler.
5. Split the data into 80% training and 20% testing sets.
6. Trained a Logistic Regression classification model.
7. Evaluated performance using precision, recall, F1-score, and ROC-AUC.
8. Exported test-set churn probabilities and risk categories.

## Model Evaluation

| Metric | Result |
|---|---:|
| Precision | 0.5763 |
| Recall | 0.5075 |
| F1-score | 0.5397 |
| ROC-AUC | 0.6808 |
| Accuracy | 64% |

## Project Files

- `churn_analysis.py` — model training and evaluation script
- `customer_churn_model_v2.csv` — dataset
- `churn_risk_predictions.csv` — test-set predictions and risk scores
- `roc_curve.png` — ROC curve
- `confusion_matrix.png` — confusion matrix
- `requirements.txt` — required Python libraries

## Limitations

The baseline model shows moderate ability to distinguish churners from non-churners. Its predictions should be improved and validated before being used for real customer-retention decisions. The risk categories are illustrative thresholds.

## How to Run

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the analysis:

```bash
python churn_analysis.py
```

## Conclusion

A Logistic Regression pipeline was developed to estimate customer churn risk. The project demonstrates data preprocessing, classification, model evaluation, visualization, and prediction export.

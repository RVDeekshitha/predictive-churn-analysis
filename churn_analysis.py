
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

# 1. Load the dataset
df = pd.read_csv("customer_churn_model_v2.csv")

print("Dataset loaded successfully:", df.shape)

# 2. Separate target and input features
# CustomerID is excluded because it is only an identifier.
X = df.drop(columns=["Churn", "CustomerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in the Churn target column.")

# 3. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "str", "category", "bool"]
).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 4. Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", MinMaxScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# 6. Build and train the model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# 7. Generate test predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 8. Evaluate model performance
print("\n========== MODEL EVALUATION ==========")
print("Precision:", round(precision_score(y_test, y_pred, zero_division=0), 4))
print("Recall:   ", round(recall_score(y_test, y_pred, zero_division=0), 4))
print("F1-score: ", round(f1_score(y_test, y_pred, zero_division=0), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["No Churn", "Churn"],
    zero_division=0
))

# 9. Plot ROC curve
RocCurveDisplay.from_predictions(y_test, y_prob)
plt.title("ROC Curve - Customer Churn Prediction")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=300)
plt.show()

# 10. Plot confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    display_labels=["No Churn", "Churn"]
)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.show()

# 11. Export test-set predictions
test_results = pd.DataFrame({
    "CustomerID": df.loc[X_test.index, "CustomerID"],
    "ActualChurn": y_test.map({0: "No", 1: "Yes"}),
    "PredictedChurn": pd.Series(y_pred, index=X_test.index).map(
        {0: "No", 1: "Yes"}
    ),
    "ChurnProbability": y_prob
})

test_results["RiskCategory"] = pd.cut(
    test_results["ChurnProbability"],
    bins=[-0.001, 0.30, 0.60, 1.0],
    labels=["Low", "Medium", "High"]
)

test_results.to_csv("churn_risk_predictions.csv", index=False)

print("\nSaved churn_risk_predictions.csv")
print("Saved roc_curve.png and confusion_matrix.png")
print("\nSample predictions:")
print(test_results.head(10).to_string(index=False))

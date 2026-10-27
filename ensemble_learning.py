
# Practical Assignment 5
# Ensemble Learning and Performance Evaluation

# --------------------------------------------------
# 1. Import Required Libraries
# --------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    StackingClassifier,
    AdaBoostClassifier,
    VotingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 2. Load the Dataset
# --------------------------------------------------

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("=" * 60)
print("BREAST CANCER CLASSIFICATION DATASET")
print("=" * 60)

print("Dataset shape:", X.shape)
print("Number of samples:", len(X))
print("Number of features:", X.shape[1])

print("\nTarget class distribution:")
print(y.value_counts().sort_index())

print("\nMissing values:", X.isnull().sum().sum())

# --------------------------------------------------
# 3. Split the Dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# --------------------------------------------------
# 4. Define Base Classifiers
# --------------------------------------------------

logistic_regression = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000, random_state=42)
)

decision_tree = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

random_forest = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# --------------------------------------------------
# 5. Define Ensemble Classifiers
# --------------------------------------------------

# A. Stacking Ensemble
stacking_model = StackingClassifier(
    estimators=[
        ("lr", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42)
        )),
        ("dt", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ],
    final_estimator=LogisticRegression(
        max_iter=5000,
        random_state=42
    ),
    cv=5
)

# B. AdaBoost Ensemble
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(
        max_depth=1,
        random_state=42
    ),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# C. Voting Ensemble - Hard Voting
voting_model = VotingClassifier(
    estimators=[
        ("lr", make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=42)
        )),
        ("dt", DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        )),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ))
    ],
    voting="hard"
)

# --------------------------------------------------
# 6. Store All Six Models
# --------------------------------------------------

models = {
    "Logistic Regression": logistic_regression,
    "Decision Tree": decision_tree,
    "Random Forest": random_forest,
    "Stacking": stacking_model,
    "AdaBoost": adaboost_model,
    "Voting": voting_model
}

# --------------------------------------------------
# 7. Train and Evaluate All Models
# --------------------------------------------------

results = []
predictions = {}

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training:", name)
    print("=" * 60)

    # Train model
    model.fit(X_train, y_train)

    # Predict test data
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Weighted metrics account for both classes
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1],
            target_names=["Malignant", "Benign"],
            zero_division=0
        )
    )

# --------------------------------------------------
# 8. Create Performance Comparison Table
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy", ascending=False
).reset_index(drop=True)

print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(results_df.round(4).to_string(index=False))

# Save results as CSV
results_df.to_csv(
    "ensemble_performance_results.csv",
    index=False
)

print("\nComparison table saved as:")
print("ensemble_performance_results.csv")

# --------------------------------------------------
# 9. Plot Performance Comparison
# --------------------------------------------------

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]

results_plot = results_df.set_index("Model")[metrics]

ax = results_plot.plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1.05)
)

plt.title("Performance Comparison of Base and Ensemble Models")
plt.xlabel("Machine Learning Models")
plt.ylabel("Score")
plt.xticks(rotation=30, ha="right")
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.4)
plt.tight_layout()

plt.savefig("model_performance_comparison.png", dpi=300)
plt.show()

# --------------------------------------------------
# 10. Plot Confusion Matrices
# --------------------------------------------------

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for ax, (name, y_pred) in zip(axes, predictions.items()):

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant", "Benign"],
        yticklabels=["Malignant", "Benign"],
        ax=ax
    )

    ax.set_title(name)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

plt.suptitle("Confusion Matrices of All Six Models", fontsize=16)
plt.tight_layout()

plt.savefig("confusion_matrices.png", dpi=300)
plt.show()

# --------------------------------------------------
# 11. Identify Best Model
# --------------------------------------------------

best_model = results_df.iloc[0]

print("\n" + "=" * 60)
print("BEST MODEL BASED ON TEST ACCURACY")
print("=" * 60)

print("Model:", best_model["Model"])
print(f"Accuracy: {best_model['Accuracy']:.4f}")

print("\nExperiment completed successfully.")

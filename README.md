# Ensemble Learning and Performance Evaluation

## Practical Assignment 5 — AIML DMD

This project implements and compares three ensemble learning techniques — Stacking, AdaBoost, and Voting — against individual machine learning classifiers using the Breast Cancer Wisconsin dataset.

## Aim

To implement ensemble learning techniques using suitable base classifiers, evaluate their performance using classification metrics, and compare their results with individual machine learning models.

## Objectives

- Understand the concepts of ensemble learning.
- Implement individual machine learning classifiers.
- Implement Stacking using a meta-classifier.
- Implement AdaBoost using weak learners.
- Implement Voting using majority voting.
- Evaluate models using accuracy, precision, recall, F1-score, and confusion matrices.
- Visualize and compare the performance of all models.

## Dataset Description

**Dataset:** Breast Cancer Wisconsin Dataset

**Source:** Scikit-learn built-in dataset.

| Property | Description |
|---|---|
| Total samples | 569 |
| Input features | 30 |
| Problem type | Binary classification |
| Class 0 | Malignant |
| Class 1 | Benign |
| Missing values | None in the original dataset |

The dataset contains numerical features computed from digitized images of fine needle aspirate samples. The task is to classify samples as malignant or benign.

## Machine Learning Models

### Base Classifiers

1. **Logistic Regression:** A supervised classification algorithm that estimates class probabilities.
2. **Decision Tree:** A classifier that uses decision rules to predict class labels.
3. **Random Forest:** An ensemble of decision trees that combines their predictions.

### Ensemble Classifiers

**1. Stacking Ensemble**

Combines Logistic Regression, Decision Tree, and Random Forest base classifiers using a Logistic Regression meta-classifier. Five-fold cross-validation is used to generate training predictions for the meta-classifier.

**2. AdaBoost Ensemble**

Uses Decision Tree stumps as weak learners. The algorithm trains learners sequentially, emphasizing difficult-to-classify samples.

**3. Voting Ensemble**

Combines Logistic Regression, Decision Tree, and Random Forest using hard voting. The final prediction is based on the majority vote.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- VS Code

## Project Structure

```text
DMD_Practical_5_Ensemble_Learning/
│
├── ensemble_learning.py
├── README.md
├── ensemble_performance_results.csv
├── model_performance_comparison.png
├── confusion_matrices.png
└── accuracy_comparison.png       # Optional
```

The
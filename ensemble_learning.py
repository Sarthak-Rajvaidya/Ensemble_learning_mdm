# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

# Load dataset
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Display dataset information
print("Dataset Shape:", X.shape)
print("\nFirst 5 rows:")
print(X.head())

print("\nTarget classes:")
print(y.value_counts())

print("\nMissing values:", X.isnull().sum().sum())
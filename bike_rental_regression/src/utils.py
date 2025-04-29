# utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

def load_data(filepath):
    return pd.read_csv(filepath)

def plot_relationships(df):
    sns.pairplot(df)
    plt.show()

def plot_correlation_matrix(df):
    plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.show()

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R² Score: {r2:.2f}')

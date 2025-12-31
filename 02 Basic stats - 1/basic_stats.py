"""
Assignment 2: Basic Statistics
Descriptive Analytics and Data Preprocessing on Sales & Discounts Dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
def load_data():
    """Load the sales data"""
    df = pd.read_csv('sales_data_with_discounts.csv')
    return df


# 1. Descriptive Analytics
def descriptive_stats(df):
    """Calculate mean, median, mode, and standard deviation"""
    print("=== Descriptive Statistics ===\n")

    # Get numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Mean: {df[col].mean():.2f}")
        print(f"  Median: {df[col].median():.2f}")
        print(f"  Mode: {df[col].mode()[0]:.2f}")
        print(f"  Standard Deviation: {df[col].std():.2f}")

    print("\n" + "="*50 + "\n")


# 2. Data Visualization - Histograms
def plot_histograms(df):
    """Plot histograms for numerical columns"""
    print("=== Creating Histograms ===\n")

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    # Create subplots
    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        axes[i].hist(df[col], bins=20, edgecolor='black')
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Hide extra subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('histograms.png')
    print("Histograms saved as 'histograms.png'\n")
    plt.close()


# 3. Boxplots
def plot_boxplots(df):
    """Create boxplots to identify outliers"""
    print("=== Creating Boxplots ===\n")

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    n_cols = 3
    n_rows = (len(numerical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel(col)

    # Hide extra subplots
    for i in range(len(numerical_cols), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('boxplots.png')
    print("Boxplots saved as 'boxplots.png'\n")
    plt.close()


# 4. Bar Charts for Categorical Data
def plot_bar_charts(df):
    """Create bar charts for categorical columns"""
    print("=== Creating Bar Charts ===\n")

    categorical_cols = df.select_dtypes(include=['object']).columns

    n_cols = 2
    n_rows = (len(categorical_cols) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts()
        axes[i].bar(range(len(value_counts)), value_counts.values)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')
        axes[i].set_xticks(range(len(value_counts)))
        axes[i].set_xticklabels(value_counts.index, rotation=45, ha='right')

    # Hide extra subplots
    for i in range(len(categorical_cols), len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig('bar_charts.png')
    print("Bar charts saved as 'bar_charts.png'\n")
    plt.close()


# 5. Standardization (Z-score normalization)
def standardize_data(df):
    """Standardize numerical columns using z-score"""
    print("=== Standardization ===\n")

    numerical_cols = df.select_dtypes(include=[np.number]).columns

    df_standardized = df.copy()

    for col in numerical_cols:
        mean = df[col].mean()
        std = df[col].std()
        df_standardized[col] = (df[col] - mean) / std

    print("Before Standardization:")
    print(df[numerical_cols].head())

    print("\nAfter Standardization:")
    print(df_standardized[numerical_cols].head())

    print("\n" + "="*50 + "\n")

    return df_standardized


# 6. One-Hot Encoding
def one_hot_encoding(df):
    """Convert categorical variables to dummy variables"""
    print("=== One-Hot Encoding ===\n")

    categorical_cols = df.select_dtypes(include=['object']).columns

    print("Original dataset shape:", df.shape)
    print("\nCategorical columns:", list(categorical_cols))

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    print("\nAfter One-Hot Encoding:")
    print("New dataset shape:", df_encoded.shape)
    print("\nFirst few rows:")
    print(df_encoded.head())

    print("\n" + "="*50 + "\n")

    return df_encoded


# Main function
def main():
    # Load data
    df = load_data()

    print("Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}\n")
    print("="*50 + "\n")

    # 1. Descriptive Statistics
    descriptive_stats(df)

    # 2. Histograms
    plot_histograms(df)

    # 3. Boxplots
    plot_boxplots(df)

    # 4. Bar Charts
    plot_bar_charts(df)

    # 5. Standardization
    df_standardized = standardize_data(df)

    # 6. One-Hot Encoding
    df_encoded = one_hot_encoding(df)

    print("All analyses completed successfully!")


if __name__ == "__main__":
    main()

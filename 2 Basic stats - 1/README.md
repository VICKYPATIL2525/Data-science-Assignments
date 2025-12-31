# Assignment 2: Basic Statistics

## Topics Covered

### 1. Pandas Library
- `pd.read_csv()` - load CSV data
- `df.select_dtypes()` - select columns by data type
- `df.head()` - view first rows
- `df.shape` - get dimensions (rows, columns)
- `pd.get_dummies()` - one-hot encoding

### 2. Descriptive Statistics
- `mean()` - average value
- `median()` - middle value
- `mode()` - most frequent value
- `std()` - standard deviation

### 3. Data Visualization
- **Matplotlib** - `plt.hist()`, `plt.boxplot()`, `plt.bar()`
- **Histograms** - show distribution
- **Boxplots** - identify outliers
- **Bar Charts** - categorical data frequency

### 4. Data Preprocessing
- **Standardization** (Z-score normalization): `z = (x - mean) / std`
- **One-Hot Encoding** - convert categories to binary columns

### 5. NumPy
- `np.number` - numerical data types
- Array operations

### 6. Data Types
- Numerical: int, float
- Categorical: object (strings)

---

## Interview Questions

### Q1: What is the difference between mean, median, and mode?
```python
data = [1, 2, 2, 3, 4, 5, 100]

# Mean - average (affected by outliers)
mean = sum(data) / len(data)  # 16.7

# Median - middle value (robust to outliers)
median = 3  # middle value when sorted

# Mode - most frequent
mode = 2  # appears twice
```

### Q2: What is standard deviation?
Measures how spread out the data is from the mean.
- Low std → data is close to mean
- High std → data is spread out

```python
import numpy as np
data = [1, 2, 3, 4, 5]
std = np.std(data)  # 1.41
```

### Q3: How to select numerical columns in pandas?
```python
numerical_cols = df.select_dtypes(include=[np.number]).columns
categorical_cols = df.select_dtypes(include=['object']).columns
```

### Q4: What is standardization (z-score)?
Scales data to have mean=0 and std=1.

**Formula:** `z = (x - mean) / std`

```python
# Manual
mean = df['column'].mean()
std = df['column'].std()
df['column_standardized'] = (df['column'] - mean) / std

# Using sklearn
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```

### Q5: Why standardize data?
- Makes different features comparable (same scale)
- Required for algorithms like KNN, SVM, Neural Networks
- Improves convergence in gradient descent

### Q6: What is one-hot encoding?
Converts categorical variables to binary columns.

```python
df = pd.DataFrame({'color': ['red', 'blue', 'red', 'green']})

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['color'])

# Result:
#    color_red  color_blue  color_green
# 0      1          0           0
# 1      0          1           0
# 2      1          0           0
# 3      0          0           1
```

### Q7: What is the difference between `drop_first=True` and `drop_first=False`?
```python
# drop_first=False - creates column for every category
pd.get_dummies(df, drop_first=False)  # n columns

# drop_first=True - drops first category (avoid multicollinearity)
pd.get_dummies(df, drop_first=True)   # n-1 columns
```

### Q8: What are outliers and how to detect them?
Outliers are extreme values that differ significantly from other observations.

**Detection methods:**
- **Boxplot** - values beyond 1.5 * IQR
- **Z-score** - |z| > 3
- **IQR method**

```python
Q1 = df['column'].quantile(0.25)
Q3 = df['column'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['column'] < lower_bound) | (df['column'] > upper_bound)]
```

### Q9: What is a histogram?
Shows frequency distribution of numerical data.

```python
import matplotlib.pyplot as plt

plt.hist(df['column'], bins=20)
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.show()
```

### Q10: What is a boxplot?
Shows data distribution with quartiles and outliers.

**Components:**
- Box: Q1 to Q3 (IQR)
- Line in box: Median
- Whiskers: 1.5 * IQR
- Dots: Outliers

```python
plt.boxplot(df['column'])
plt.show()
```

### Q11: Difference between `.loc[]` and `.iloc[]`?
```python
# .loc - label-based (by name)
df.loc[0, 'column_name']

# .iloc - position-based (by index)
df.iloc[0, 2]
```

### Q12: How to handle missing values?
```python
# Check missing values
df.isnull().sum()

# Drop rows with missing values
df.dropna()

# Fill with mean
df['column'].fillna(df['column'].mean())

# Fill with median
df['column'].fillna(df['column'].median())

# Fill with mode
df['column'].fillna(df['column'].mode()[0])
```

### Q13: What is the difference between `copy()` and view in pandas?
```python
# Copy - creates new object
df_copy = df.copy()
df_copy['col'] = 0  # doesn't affect original

# View - references original
df_view = df['col']
df_view[0] = 999  # affects original
```

### Q14: How to read CSV files?
```python
# Basic
df = pd.read_csv('file.csv')

# With options
df = pd.read_csv('file.csv',
                 sep=',',           # delimiter
                 header=0,          # row for column names
                 index_col=0,       # column for index
                 na_values=['NA'])  # values to treat as NaN
```

### Q15: How to save DataFrame to CSV?
```python
df.to_csv('output.csv', index=False)
```

### Q16: What is `value_counts()`?
Counts unique values in a column.

```python
df['column'].value_counts()

# With percentages
df['column'].value_counts(normalize=True)
```

### Q17: Difference between normalization and standardization?
**Standardization (Z-score):**
- `z = (x - mean) / std`
- Mean = 0, Std = 1
- Range: -∞ to +∞

**Normalization (Min-Max):**
- `x_norm = (x - min) / (max - min)`
- Range: 0 to 1

```python
# Normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_normalized = scaler.fit_transform(df)
```

### Q18: How to create subplots?
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

axes[0, 0].hist(data1)
axes[0, 1].boxplot(data2)
axes[1, 0].bar(x, y)
axes[1, 1].plot(x, y)

plt.tight_layout()
plt.show()
```

### Q19: What is skewness?
Measure of asymmetry in distribution.

- **Positive skew** - tail on right (mean > median)
- **Negative skew** - tail on left (mean < median)
- **Zero skew** - symmetric (mean = median)

```python
df['column'].skew()
```

### Q20: Common pandas operations?
```python
# Shape
df.shape  # (rows, columns)

# Info
df.info()  # column types, non-null counts

# Describe
df.describe()  # statistical summary

# Unique values
df['column'].nunique()

# Sort
df.sort_values('column', ascending=False)

# Filter
df[df['column'] > 10]

# Group by
df.groupby('category')['value'].mean()
```

---

## Files
- `basic_stats.py` - Complete solution with all visualizations
- `sales_data_with_discounts.csv` - Dataset
- `Basic statistics.docx` - Assignment questions

## Required Libraries
```python
pip install pandas numpy matplotlib seaborn
```

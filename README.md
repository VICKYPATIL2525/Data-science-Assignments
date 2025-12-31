# Data Science Assignments

A comprehensive collection of **20 Data Science assignments** covering fundamental to advanced topics in Machine Learning, Statistics, and Data Analysis.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Assignments](#assignments)
  - [01 - Basics of Python](#01---basics-of-python)
  - [02 - Basic Statistics 1](#02---basic-statistics-1)
  - [03 - Basic Statistics 2](#03---basic-statistics-2)
  - [04 - EDA 1](#04---eda-1)
  - [05 - EDA 2](#05---eda-2)
  - [06 - Hypothesis Testing](#06---hypothesis-testing)
  - [07 - Decision Tree](#07---decision-tree)
  - [08 - KNN](#08---knn)
  - [09 - Logistic Regression](#09---logistic-regression)
  - [10 - SVM](#10---svm)
  - [11 - NLP and Naive Bayes](#11---nlp-and-naive-bayes)
  - [12 - Clustering](#12---clustering)
  - [13 - PCA](#13---pca)
  - [14 - Random Forest](#14---random-forest)
  - [15 - Multiple Linear Regression](#15---multiple-linear-regression)
  - [16 - Neural Networks](#16---neural-networks)
  - [17 - XGBoost and LightGBM](#17---xgboost-and-lightgbm)
  - [18 - Time Series Analysis](#18---time-series-analysis)
  - [19 - Association Rules](#19---association-rules)
  - [20 - Recommendation System](#20---recommendation-system)
- [Libraries Used](#libraries-used)
- [How to Use](#how-to-use)

---

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Running Notebooks
```bash
jupyter notebook
```

---

## Assignments

### 01 - Basics of Python
**Folder:** `01 Basics of python`

**Topics Covered:**
- Python fundamentals
- Functions and control flow
- String manipulation
- Data structures

**What's Inside:**
| Exercise | Description | Concepts |
|----------|-------------|----------|
| Prime Checker | Check if a number is prime | Loops, Conditionals |
| Multiplication Quiz | Random number product game | `random` module, User input |
| Even Squares | Print squares of even numbers (100-200) | `range()`, Modulo operator |
| Word Counter | Count frequency of each word | Dictionaries, `split()` |
| Palindrome Checker | Check if string reads same backwards | String slicing, `isalnum()` |

**Key Functions:** `is_prime()`, `is_palindrome()`

---

### 02 - Basic Statistics 1
**Folder:** `02 Basic stats - 1`

**Dataset:** Sales Data with Discounts

**Topics Covered:**
- Descriptive Statistics
- Data Visualization
- Data Preprocessing

**Technical Implementation:**
| Task | Method | Output |
|------|--------|--------|
| Central Tendency | `mean()`, `median()`, `mode()` | Statistical measures |
| Dispersion | `std()`, IQR calculation | Spread of data |
| Visualization | `matplotlib` histograms, boxplots | Distribution plots |
| Standardization | Z-score: `(x - mean) / std` | Normalized data |
| Encoding | `pd.get_dummies()` | One-Hot encoded features |

**Formulas Used:**
- **Mean:** `Σx / n`
- **Standard Deviation:** `√(Σ(x - μ)² / n)`
- **Z-score:** `(x - μ) / σ`

---

### 03 - Basic Statistics 2
**Folder:** `03 Basic stats - 2`

**Dataset:** Print-head Durability Data (15 samples)

**Topics Covered:**
- Confidence Intervals
- t-distribution vs z-distribution
- Statistical Inference

**Technical Implementation:**
| Task | Distribution | When to Use |
|------|--------------|-------------|
| Task A | t-distribution | Population σ unknown, small sample |
| Task B | z-distribution | Population σ known |

**Formulas Used:**
- **Standard Error:** `σ / √n`
- **Confidence Interval:** `x̄ ± (critical_value × SE)`
- **t-critical:** `scipy.stats.t.ppf(1 - α/2, df)`
- **z-critical:** `scipy.stats.norm.ppf(1 - α/2)`

**Key Insight:** t-distribution gives wider intervals due to heavier tails

---

### 04 - EDA 1
**Folder:** `04 EDA1`

**Dataset:** Cardiotocographic Data (Fetal Health Monitoring)

**Topics Covered:**
- Exploratory Data Analysis
- Data Cleaning
- Correlation Analysis
- Outlier Detection

**Technical Implementation:**
| Analysis | Method | Purpose |
|----------|--------|---------|
| Missing Values | `fillna(median)` | Handle nulls |
| Duplicates | `drop_duplicates()` | Remove redundancy |
| Distribution | Histograms, Boxplots | Understand spread |
| Correlation | `df.corr()`, Heatmap | Find relationships |
| Outliers | IQR method: `Q1 - 1.5*IQR`, `Q3 + 1.5*IQR` | Detect anomalies |

**Visualizations Created:**
- Histograms for all numerical features
- Boxplots for outlier detection
- Correlation heatmap
- Violin plots by fetal state (NSP)

**Target Variable:** NSP (1=Normal, 2=Suspect, 3=Pathological)

---

### 05 - EDA 2
**Folder:** `05 EDA2`

**Dataset:** Adult Census Income (~32K records)

**Topics Covered:**
- Feature Engineering
- Scaling Techniques
- Encoding Methods
- Outlier Detection with ML

**Technical Implementation:**
| Technique | Method | Use Case |
|-----------|--------|----------|
| Standard Scaling | `(x - mean) / std` | SVM, Logistic Regression |
| Min-Max Scaling | `(x - min) / (max - min)` | Neural Networks, KNN |
| Label Encoding | Map categories to integers | Tree-based models |
| One-Hot Encoding | `pd.get_dummies()` | Linear models |
| Outlier Detection | `IsolationForest(contamination=0.05)` | Anomaly removal |

**Feature Engineering:**
- `total_capital = capital_gain - capital_loss`
- `age_group` (Young/Middle/Senior/Elder)
- `capital_gain_log` (log transformation for skewness)

---

### 06 - Hypothesis Testing
**Folder:** `06 Hypothesis testing`

**Topics Covered:**
- One-sample Z-test
- Chi-Square Test of Independence
- P-values and Critical Values

**Problem 1: Z-Test (Operating Costs)**
| Component | Value |
|-----------|-------|
| H₀ | μ = 4000 (costs equal theoretical) |
| H₁ | μ > 4000 (costs higher) |
| Test Type | Right-tailed |
| Formula | `Z = (x̄ - μ) / (σ / √n)` |

**Problem 2: Chi-Square Test (Device Satisfaction)**
| Component | Value |
|-----------|-------|
| H₀ | Variables are independent |
| H₁ | Variables are associated |
| Formula | `χ² = Σ(O - E)² / E` |
| Expected | `(Row Total × Col Total) / Grand Total` |
| df | `(rows - 1) × (cols - 1)` |

**Decision Rule:** Reject H₀ if p-value < α (0.05)

---

### 07 - Decision Tree
**Folder:** `07 Decision Tree`

**Dataset:** Heart Disease Prediction

**Topics Covered:**
- Decision Tree Classification
- Hyperparameter Tuning
- Model Evaluation

**Technical Implementation:**
| Step | Method |
|------|--------|
| Encoding | `LabelEncoder()` for categorical |
| Split | 80-20 train-test |
| Model | `DecisionTreeClassifier()` |
| Tuning | Grid search over `max_depth`, `min_samples_split`, `criterion` |

**Hyperparameters Tuned:**
| Parameter | Values Tested | Effect |
|-----------|---------------|--------|
| `max_depth` | 3, 5, 7, 10, None | Controls tree complexity |
| `min_samples_split` | 2, 5, 10 | Prevents overfitting |
| `criterion` | gini, entropy | Splitting measure |

**Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

### 08 - KNN
**Folder:** `08 KNN`

**Dataset:** Zoo Animal Classification (7 animal types)

**Topics Covered:**
- K-Nearest Neighbors
- Distance Metrics
- Optimal K Selection

**Technical Implementation:**
| Step | Method |
|------|--------|
| Scaling | `StandardScaler()` (critical for KNN) |
| K Selection | Test K=1 to 20, plot accuracy |
| Distance | Euclidean, Manhattan, Minkowski |

**Distance Metrics:**
| Metric | Formula | Best For |
|--------|---------|----------|
| Euclidean | `√Σ(x₁-x₂)²` | Continuous data |
| Manhattan | `Σ|x₁-x₂|` | Grid-like paths |
| Minkowski | `(Σ|x₁-x₂|^p)^(1/p)` | Generalized |

**Output:** Decision boundary visualization

---

### 09 - Logistic Regression
**Folder:** `09 Logistic Regression`

**Dataset:** Titanic Survival Prediction

**Topics Covered:**
- Binary Classification
- Sigmoid Function
- Model Interpretation

**Technical Implementation:**
| Step | Method |
|------|--------|
| Missing Values | Age → median, Embarked → mode |
| Encoding | Sex (male=1, female=0), Embarked (S=0, C=1, Q=2) |
| Scaling | `StandardScaler()` |
| Model | `LogisticRegression(max_iter=1000)` |

**Features Used:** Pclass, Sex, Age, SibSp, Parch, Fare, Embarked

**Coefficient Interpretation:**
- **Positive coefficient** → Increases survival probability
- **Negative coefficient** → Decreases survival probability
- Sex (negative): Being male reduces survival chances

**Output:** Model saved as `logistic_model.pkl` for deployment

---

### 10 - SVM
**Folder:** `10 SVM`

**Dataset:** Mushroom Classification (Edible vs Poisonous)

**Topics Covered:**
- Support Vector Machines
- Kernel Functions
- Hyperparameter Tuning

**Kernels Compared:**
| Kernel | Description | Use Case |
|--------|-------------|----------|
| Linear | Straight hyperplane | Linearly separable |
| RBF | Gaussian radial basis | Non-linear, default |
| Polynomial | Polynomial decision boundary | Complex patterns |
| Sigmoid | Tanh-like function | Neural network similarity |

**Hyperparameters Tuned:**
| Parameter | Values | Effect |
|-----------|--------|--------|
| C | 0.1, 1, 10, 100 | Regularization strength |
| gamma | scale, auto, 0.1, 1 | Kernel coefficient |

**Note:** SVM is computationally expensive for large datasets

---

### 11 - NLP and Naive Bayes
**Folder:** `11 NLP and Naive Bayes`

**Dataset:** Blog Posts (4.48 MB - Large dataset)

**Topics Covered:**
- Text Preprocessing
- TF-IDF Vectorization
- Naive Bayes Classification
- Sentiment Analysis

**Text Preprocessing Pipeline:**
```
Raw Text → Lowercase → Remove URLs → Remove Punctuation →
Remove Numbers → Remove Stopwords → Clean Text
```

**Technical Implementation:**
| Step | Method |
|------|--------|
| Tokenization | `text.split()` |
| Stopwords | `nltk.corpus.stopwords` |
| Vectorization | `TfidfVectorizer(max_features=5000)` |
| Classification | `MultinomialNB()` |
| Sentiment | `TextBlob(text).sentiment.polarity` |

**Sentiment Categories:**
- Polarity > 0.1 → Positive
- Polarity < -0.1 → Negative
- Otherwise → Neutral

---

### 12 - Clustering
**Folder:** `12 Clustering`

**Dataset:** EastWest Airlines Customer Data

**Topics Covered:**
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN

**Algorithms Compared:**
| Algorithm | Method | Key Parameter |
|-----------|--------|---------------|
| K-Means | Centroid-based | `n_clusters` (Elbow method) |
| Hierarchical | Agglomerative | `linkage` (ward) |
| DBSCAN | Density-based | `eps`, `min_samples` |

**Evaluation:** Silhouette Score (higher = better separation)

**Visualizations:**
- Elbow curve for optimal K
- Dendrogram for hierarchical
- Scatter plots with cluster colors

---

### 13 - PCA
**Folder:** `13 PCA`

**Dataset:** Wine Quality

**Topics Covered:**
- Principal Component Analysis
- Dimensionality Reduction
- Variance Explained

**Technical Implementation:**
| Step | Method |
|------|--------|
| Scaling | `StandardScaler()` (required for PCA) |
| PCA | `PCA(n_components=2)` |
| Variance | `pca.explained_variance_ratio_` |

**Output:**
- Scree plot (variance per component)
- Cumulative variance plot
- 2D scatter plot of transformed data
- K-Means clustering on PCA data

**Rule of Thumb:** Keep components explaining 95% variance

---

### 14 - Random Forest
**Folder:** `14 Random Forest`

**Dataset:** Glass Identification

**Topics Covered:**
- Ensemble Learning (Bagging)
- Random Forest vs Decision Tree
- Feature Importance

**Technical Implementation:**
| Model | Method | Advantage |
|-------|--------|-----------|
| Decision Tree | Single tree | Interpretable |
| Random Forest | 100 trees (bagging) | Reduces overfitting |

**Key Concept:** Random Forest averages multiple trees to reduce variance

**Feature Importance:** Based on Gini importance across all trees

---

### 15 - Multiple Linear Regression
**Folder:** `15 MLR`

**Dataset:** Toyota Corolla Prices

**Topics Covered:**
- Multiple Linear Regression
- Model Evaluation Metrics
- Residual Analysis

**Metrics:**
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| MSE | `Σ(y - ŷ)² / n` | Average squared error |
| RMSE | `√MSE` | Error in original units |
| MAE | `Σ|y - ŷ| / n` | Average absolute error |
| R² | `1 - (SS_res / SS_tot)` | Variance explained |

**Assumptions Checked:**
- Linearity (scatter plots)
- Normality of residuals (histogram)
- Homoscedasticity (residual vs fitted)

---

### 16 - Neural Networks
**Folder:** `16 Neural networks`

**Dataset:** Alphabet Recognition (26 classes)

**Topics Covered:**
- Multi-Layer Perceptron (MLP)
- Activation Functions
- Network Architecture

**Architecture Tested:**
| Architecture | Hidden Layers |
|--------------|---------------|
| (50,) | 1 layer, 50 neurons |
| (100,) | 1 layer, 100 neurons |
| (100, 50) | 2 layers |
| (128, 64, 32) | 3 layers (default) |

**Technical Implementation:**
```python
MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    max_iter=500
)
```

**Output:** Loss curve visualization

**Note:** Resource-intensive due to multiple architecture comparisons

---

### 17 - XGBoost and LightGBM
**Folder:** `17 XGBM & LGBM`

**Dataset:** Titanic Survival

**Topics Covered:**
- Gradient Boosting
- XGBoost vs LightGBM
- Feature Importance

**Model Comparison:**
| Model | Strength | Speed |
|-------|----------|-------|
| XGBoost | Regularization, handles missing | Moderate |
| LightGBM | Leaf-wise growth, faster | Fast |

**Common Parameters:**
```python
n_estimators=100
max_depth=5
learning_rate=0.1
```

**Output:** Feature importance comparison charts

---

### 18 - Time Series Analysis
**Folder:** `18 Timeseries`

**Dataset:** Exchange Rate Data

**Topics Covered:**
- Time Series Decomposition
- Stationarity Testing
- ARIMA Forecasting

**Technical Implementation:**
| Step | Method | Purpose |
|------|--------|---------|
| Decomposition | `seasonal_decompose()` | Trend + Seasonal + Residual |
| Stationarity | ADF Test (`adfuller()`) | Check if stationary |
| ACF/PACF | `plot_acf()`, `plot_pacf()` | Determine ARIMA orders |
| Forecasting | `ARIMA(order=(p,d,q))` | Predict future values |

**ADF Test Interpretation:**
- p-value < 0.05 → Stationary
- p-value >= 0.05 → Non-stationary (needs differencing)

---

### 19 - Association Rules
**Folder:** `19 Association Rules`

**Dataset:** Online Retail Transactions

**Topics Covered:**
- Market Basket Analysis
- Apriori Algorithm
- Support, Confidence, Lift

**Metrics Explained:**
| Metric | Formula | Meaning |
|--------|---------|---------|
| Support | `P(A ∩ B)` | How often items appear together |
| Confidence | `P(B|A) = P(A ∩ B) / P(A)` | Probability of B given A |
| Lift | `Confidence / P(B)` | How much more likely |

**Technical Implementation:**
```python
# Create basket matrix
basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack()

# Find frequent itemsets
frequent_itemsets = apriori(basket_encoded, min_support=0.02)

# Generate rules
rules = association_rules(frequent_itemsets, metric='lift', min_threshold=1.0)
```

**Business Use:** Product placement, cross-selling, bundle offers

---

### 20 - Recommendation System
**Folder:** `20 Recommendation System`

**Dataset:** Anime Ratings

**Topics Covered:**
- Content-Based Filtering
- Collaborative Filtering Concepts
- Similarity Measures

**Technical Implementation:**
| Method | Description |
|--------|-------------|
| TF-IDF | Convert genres to vectors |
| Cosine Similarity | `cosine_similarity(tfidf_matrix)` |
| Recommendation | Find top N similar items |

**Recommendation Function:**
```python
def get_recommendations(title, n=10):
    idx = indices[title]
    sim_scores = cosine_similarity[idx]
    top_indices = sim_scores.argsort()[-n-1:-1][::-1]
    return df['name'].iloc[top_indices]
```

**Types of Recommenders:**
1. **Content-Based:** Uses item features (genre, description)
2. **Collaborative:** Uses user-item interactions
3. **Hybrid:** Combines both approaches

---

## Libraries Used

| Category | Libraries |
|----------|-----------|
| Data Processing | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Gradient Boosting | `xgboost`, `lightgbm` |
| NLP | `nltk`, `textblob` |
| Time Series | `statsmodels` |
| Association Rules | `mlxtend` |
| Statistics | `scipy` |

---

## How to Use

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd Data-science-Assignments
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to any assignment folder**
   ```bash
   cd "01 Basics of python"
   ```

4. **Open Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

5. **Run cells sequentially** (Shift + Enter)

---

## Author

Data Science Learning Journey

---

## License

For educational purposes only.

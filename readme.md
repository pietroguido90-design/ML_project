# 🍷 Wine Quality AI Predictor

An end-to-end Machine Learning pipeline designed to predict wine quality from physicochemical laboratory properties using domain-driven feature engineering and a decoupled dual-model Random Forest architecture.

---

## 📌 Executive Summary

Determining wine quality through sensory tasting panels is time-consuming, expensive, and inherently subjective. This project provides an automated, objective pre-certification framework for wineries using 12 physicochemical parameters (such as acidity, residual sugar, and sulfur dioxide levels).

To maximize commercial reliability, the core objective of this project was to **minimize False Positives** (preventing ordinary table wines from being miscertified as premium offerings) while maintaining high predictive coverage.

### Key Results
* **Overall Accuracy:** `90.0%`
* **Precision (High Quality):** `74.0%`
* **Recall (High Quality):** `77.0%`
* **Predictive Power (Combined ROC-AUC):** `93.6%`
* **False Positive Reduction:** Only `70` false alarms out of `1,045` standard test samples.

---

## 📊 Dataset & Domain Challenges

The project utilizes the **UCI Wine Quality Dataset** (*Cortez et al., 2009*), comprising physicochemical properties and sensory evaluation scores for Portuguese *Vinho Verde* wines.

* **Total Samples:** 6,497 wines (4,898 White, 1,599 Red)
* **Target Classification:** Binary Target (`quality >= 7` mapped to `1` [High Quality], `< 7` mapped to `0` [Standard Quality]).

### Core Data Challenges
1. **Severe Class Imbalance:** Premium wines represent only **19.6%** of the dataset (dropping to just **6.3%** among red wines).
2. **Category Asymmetry:** White wines outnumber red wines by a 3:1 ratio.
3. **Cross-Category Noise:** Red and white wines exhibit fundamental chemical differences (e.g., volatile acidity thresholds and total sulfur dioxide ranges), causing unified single-model classifiers to bias toward white wine patterns.

---

## 🧪 Enological Feature Engineering

To provide the ensemble trees with stronger decision boundaries beyond raw, independent lab attributes, 5 domain-specific chemical ratios were engineered based on enological principles:

1. **Total Acidity:**
   $$\text{Total Acidity} = \text{fixed\_acidity} + \text{volatile\_acidity} + \text{citric\_acid}$$
2. **Bound Sulfur Dioxide:**
   $$\text{Bound } \text{SO}_2 = \text{total\_sulfur\_dioxide} - \text{free\_sulfur\_dioxide}$$
3. **SO2 Ratio (Oxidation Protection Index):**
   $$\text{Sulfur Ratio} = \frac{\text{free\_sulfur\_dioxide}}{\text{total\_sulfur\_dioxide} + 10^{-5}}$$
4. **Sugar-to-Alcohol Index (Fermentation Completeness):**
   $$\text{Sugar Alcohol Ratio} = \frac{\text{residual\_sugar}}{\text{alcohol} + 10^{-5}}$$
5. **Acid-to-Sugar Balance:**
   $$\text{Acid Sugar Ratio} = \frac{\text{Total Acidity}}{\text{residual\_sugar} + 10^{-5}}$$

---

## 🏗️ Model Architecture & Key Innovations

### Decoupled Dual-Model Strategy (The Winning Innovation)
Instead of training a single unified model with a binary `type` flag, the pipeline decouples input data into two dedicated **Random Forest Classifiers**:
* **White Wine Random Forest:** Optimized for higher sulfur dioxide variance, citric acid levels, and sugar ranges.
* **Red Wine Random Forest:** Optimized for volatile acidity sensitivity, higher sulphate concentrations, and dense acid profiles.

```
                    [ Input Sample ]
                           │
                 Is Sample Red or White?
                 ┌─────────┴─────────┐
                 ▼                   ▼
         [ Red Wine RF ]     [ White Wine RF ]
                 │                   │
                 └─────────┬─────────┘
                           ▼
              [ Predicted Quality Label ]
```

### Why Oversampling (SMOTE) Was Rejected
During experimentation, synthetic oversampling (SMOTE) was applied to the training set. While it slightly increased Recall ($67.6\% \rightarrow 70.7\%$), it introduced artificial boundary noise that dropped Precision from **$70.6\%$ down to $66.0\%$** and increased False Positives from 72 to 95. To protect commercial credibility, SMOTE was discarded in favor of class-weighted tree splits.

---

## 📈 Benchmark & Performance Comparison

Evaluated on an independent, stratified 20% holdout Test Set ($N = 1,300$ total test samples: 980 White, 320 Red):

| Model Architecture | High-Quality Precision | High-Quality Recall | F1-Score | Overall Accuracy | Predictive Power (ROC-AUC) | False Positives (FP) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $38.2\%$ | $75.0\%$ | $0.51$ | $74.5\%$ | $80.2\%$ | 310 |
| **Decision Tree** | $41.2\%$ | $80.1\%$ | $0.54$ | $76.2\%$ | $81.2\%$ | 293 |
| **XGBoost (Unified)** | $64.3\%$ | $68.4\%$ | $0.66$ | $86.5\%$ | $90.2\%$ | 97 |
| **Random Forest (Unified Base)** | $70.6\%$ | $67.6\%$ | $0.69$ | $88.2\%$ | $91.0\%$ | 72 |
| **RF + SMOTE** | $66.0\%$ | $70.7\%$ | $0.68$ | $87.0\%$ | $91.4\%$ | 95 |
| **Decoupled Dual RF + Feature Eng. (FINAL)** | **74.0%** | **77.0%** | **0.75** | **90.0%** | **93.6%** | **70** |

### Individual Sub-Model Breakdown
* **White Wine Model:** Precision: `74%` | Recall: `78%` | ROC-AUC: `0.9319`
* **Red Wine Model:** Precision: `74%` | Recall: `72%` | ROC-AUC: `0.9451` *(Record high)*

---

## 🛠️ Project Structure

```text
├── data/
│   ├── raw/                  # Original UCI CSV files (winequality-red.csv, winequality-white.csv)
│   └── processed/            # Cleaned & merged datasets
├── notebooks/
│   └── wine_quality_ml.ipynb # Complete EDA, Feature Engineering, & Training Pipeline
├── src/
│   ├── feature_engineering.py# Domain ratio generator script
│   └── train_predict.py      # Decoupled Dual Random Forest pipeline
├── README.md                 # Project Documentation
└── requirements.txt          # PythonDependencies
```

---

## 🚀 How to Run

### 1. Requirements
Ensure you have Python 3.9+ installed along with the required libraries:
```bash
pip install pandas numpy scikit-learn imbalanced-learn
```

### 2. Execution
Run the complete pipeline from notebook or Python environment:
```python
from sklearn.ensemble import RandomForestClassifier

# Instantiate White Wine Model
rf_white = RandomForestClassifier(
    n_estimators=300, max_depth=None, class_weight='balanced', random_state=42
)

# Instantiate Red Wine Model
rf_red = RandomForestClassifier(
    n_estimators=300, max_depth=None, class_weight='balanced', random_state=42
)

# Fit decoupled models
rf_white.fit(X_train_white, y_train_white)
rf_red.fit(X_train_red, y_train_red)
```

---

## 📄 License & Acknowledgments

* **Dataset Source:** [UCI Machine Learning Repository - Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
* **Citation:** Cortez, P., Cerdeira, A., Almeida, F., Matos, T., & Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties*. Decision Support Systems, 47(4), 547-553.
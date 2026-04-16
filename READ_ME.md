# Predictive Maintenance & Anomaly Detection in Industrial Machinery

A machine learning project that predicts equipment failures and detects anomalies 
in industrial machinery sensor data — helping reduce downtime and maintenance costs.

---

## Research Questions

1. Which machine learning algorithms perform best for predictive maintenance?
2. Can machine faults be anticipated before they occur using early-warning patterns?
3. Which failure type is most likely to occur given current operating conditions?
4. Are there quantifiable anomalies in machine signals that appear prior to failure?

---

## Goal

To build a proactive maintenance system that identifies machine faults early, 
classifies failure types, and reduces unplanned downtime and operational costs 
in industrial environments.

---

## Dataset

**Source:** [Machine Predictive Maintenance Classification – Kaggle](https://www.kaggle.com/datasets/shivamb/machine-predictive-maintenance-classification)

- 5 distinct failure categories + binary failure target
- Realistic sensor readings: air temperature, process temperature, rotational 
  speed, torque, and tool wear
- Contains natural industrial anomalies (not noise)
- SMOTE oversampling applied to handle class imbalance

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![Pandas](https://img.shields.io/badge/Pandas-Data-lightgrey)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Viz-yellow)

- Python, pandas, NumPy
- scikit-learn (Logistic Regression, Random Forest, Decision Tree, 
  Isolation Forest, LOF)
- XGBoost
- Matplotlib, Seaborn
- imbalanced-learn (SMOTE)

---

## Project Pipeline

### 1. Exploratory Data Analysis (EDA)
- Dataset overview, structure, and summary statistics
- Feature distribution visualisation (skewness, outliers)
- Failure category distribution analysis
- Correlation heatmaps across numerical variables
- Boxplots comparing sensor readings across failure types

### 2. Data Preprocessing
- Missing value inspection
- Removal of irrelevant columns
- Duplicate detection and removal
- Outlier inspection (distinguishing noise from real anomalies)
- Data type correction
- Feature encoding and scaling
- Train/test splitting

### 3. Model Training & Evaluation

#### Binary Classification (Failure / No Failure)
| Model | Notes |
|---|---|
| Logistic Regression | Strong baseline |
| XGBoost | Best overall — captures nonlinear patterns |

#### Multi-Class Classification (Failure Type)
| Model | Notes |
|---|---|
| Decision Tree | Interpretable baseline |
| Random Forest | Best overall — consistent across all 5 failure types |

#### Anomaly Detection (Unsupervised)
| Model | Notes |
|---|---|
| Isolation Forest | Flags unusual machine behaviour |
| Local Outlier Factor (LOF) | Detects early-stage faults not visible in labels |

---

## Key Findings

- **XGBoost** outperformed Logistic Regression for binary failure prediction due 
  to its ability to model complex feature interactions
- **Random Forest** delivered the most consistent results for multi-class failure 
  classification across all five failure categories
- **Anomaly detection** (Isolation Forest & LOF) successfully identified 
  early-stage faults not captured by supervised labels
- SMOTE oversampling improved model robustness on the imbalanced dataset
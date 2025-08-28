# Heart Attack Prediction 🫀

[![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-passing-brightgreen)](https://github.com/balajiabcd/Heart-Attact-Prediction/actions)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Kaggle-red)](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Data Analysis & Visualizations](#data-analysis--visualizations)
4. [Model Development](#model-development)
5. [Web Application Deployment](#web-application-deployment)
6. [Installation & Setup](#installation--setup)
7. [Project Structure](#project-structure)
8. [Running Tests](#running-tests)
9. [Notes](#notes)   


## Project Overview
The **Heart Attack Prediction** project is a machine learning-based solution designed to predict the likelihood of heart attacks in patients using clinical and diagnostic data. Accurate prediction of heart attack risk is crucial for early intervention, preventive care, and informed medical decision-making.

Key highlights of the project:

- **Objective:** Classify patients into high-risk and low-risk categories for heart attacks.  
- **Approach:** Uses structured patient data including both **quantitative variables** (e.g., age, cholesterol, blood pressure) and **qualitative variables** (e.g., chest pain type, gender).  
- **Workflow:**  
  1. **Data preprocessing** – handle missing values, encode categorical features, and standardize numerical variables.  
  2. **Exploratory Data Analysis (EDA)** – visualize distributions, correlations, and relationships among features.  
  3. **Feature engineering** – create meaningful features, interaction terms, and risk scores.  
  4. **Model development** – train multiple classifiers and select the best-performing model based on evaluation metrics.  
  5. **Deployment** – integrate the final model into a web application hosted on **AWS** using **Flask**, with predictions accessible through a user-friendly interface.

The project demonstrates end-to-end implementation, from data exploration to model deployment, making it a practical reference for healthcare analytics applications.

---

## Dataset
The dataset used in this project is publicly available on **Kaggle** and contains **303 patient records** with attributes relevant to cardiovascular health.  

**Dataset details:**

- **Number of records:** 303  
- **Outcome distribution:**  
  - 165 patients (high risk)  
  - 138 patients (low risk)  
- **Features:**  
  - **Quantitative:** Age, resting blood pressure, cholesterol, maximum heart rate achieved, etc.  
  - **Qualitative / Categorical:** Gender, chest pain type (cp), fasting blood sugar (fbs), resting ECG results (restecg), slope of peak exercise ST segment (slp), number of major vessels (caa), thalassemia (thall), exercise-induced angina (exng).  

**Key preprocessing steps:**

- **Handling categorical variables:** Features with 2–7 unique values are one-hot encoded to ensure proper model input.  
- **Feature selection:** Highly correlated or redundant features are removed to avoid multicollinearity.  
- **Data quality:** The dataset is complete with no missing values, simplifying preprocessing and ensuring model reliability.

Dataset link: [Kaggle Heart Attack Dataset](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset)

---

## Data Analysis & Visualizations
Comprehensive data exploration was conducted to understand relationships, detect patterns, and prepare for model development. Visualization played a critical role in this phase.

**1. Correlation Heatmap:**  
A heatmap of feature correlations highlights dependencies between variables. Highly correlated features were considered for removal or transformation to improve model generalization.  

![Heatmap](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/static/images/heatmap.png)

**2. Histograms:**  
Histograms for all numerical features reveal differences in distributions between patients with high and low heart attack risk. These visualizations help identify predictive features.  

![Histplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/static/images/histplot.png)

**3. Pairplots & Scatterplots:**  
Pairwise plots indicate clustering among certain feature combinations, showing potential separability between high-risk and low-risk groups. This provides an early signal that classification models could achieve high accuracy.  

![Pairplot](https://github.com/balajiabcd/Heart-Attact-Prediction/blob/main/static/images/pairplot.png)

**4. Additional Visualizations:**  
- Boxplots for feature distributions (e.g., age, cholesterol, blood pressure)  
- Interaction plots for combined features to understand non-linear relationships  
- Class distribution charts to visualize dataset balance  

All generated plots are stored in the `data_analysis` folder, enabling detailed inspection for feature relevance and model insights.



# Heart Attack Prediction

This project is a **machine learning pipeline** that predicts the likelihood of heart attacks based on patient health data. It includes data preprocessing, feature engineering, visualization, model training, evaluation, and testing.

---

## 📂 Project Structure

```
Heart-Attact-Prediction/
│
├── data/                   # Dataset files (CSV or raw data, e.g. heart.csv)
├── src/                    # Source code for ML pipeline
│   ├── data_utils.py       # Data loading, preprocessing, encoding, scaling
│   ├── feature_engineering.py # Domain-driven feature creation & PCA
│   ├── model.py            # Training & evaluation of ML models
│   ├── visualize.py        # Plots and data visualization
│
├── tests/                  # Unit tests for each module
│   ├── test_data_utils.py
│   ├── test_feature_engineering.py
│   ├── test_model.py
│   ├── test_visualisation.py
│
├── static/data_analysis/   # Auto-generated plots (EDA, correlations, etc.)
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── results/                # Model performance outputs
```

---

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/balajiabcd/Heart-Attact-Prediction.git
   cd Heart-Attact-Prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset

The dataset (`heart.csv`) contains patient health records with features like:

- Age
- Resting blood pressure
- Cholesterol levels
- Maximum heart rate achieved
- Exercise-induced angina
- ST depression
- Other derived medical indicators (risk flags, ratios, PCA features)

---

## 🚀 Usage

1. **Run data preprocessing and feature engineering**:
   ```bash
   python src/data_utils.py
   ```

2. **Train and evaluate multiple ML models**:
   ```bash
   python src/model.py
   ```

   This saves trained models in `models/` and results in `results/model_performance.csv`.

3. **Generate visualizations**:
   Plots are automatically stored in `static/data_analysis/`.

4. **Run tests**:
   ```bash
   pytest -v
   ```

---

## 📈 Results

- Models implemented: SVM, Random Forest, Decision Tree, KNN, Naive Bayes
- Metrics evaluated: Accuracy, Precision, Recall, F1-score, Confusion Matrix
- Results saved in: `results/model_performance.csv`

Example metrics:
- Accuracy: ~85% (Random Forest with tuned parameters)
- Precision/Recall balanced across classes

---

## 🛠️ Tech Stack

- Python 3.x
- pandas, NumPy
- scikit-learn
- seaborn, matplotlib
- pytest (for testing)
- joblib (for model saving)

---

## ✅ Testing

Tests are included to validate:

- Data preprocessing (`test_data_utils.py`)
- Feature engineering (`test_feature_engineering.py`)
- Model training & evaluation (`test_model.py`)
- Visualization (`test_visualisation.py`)

Run all tests with:
```bash
pytest -v
```

---

## 📌 Future Work

- Deploy as REST API or Streamlit dashboard
- Hyperparameter optimization (GridSearchCV/Optuna)
- Deep learning approaches (PyTorch/TensorFlow)
- Support larger datasets for robustness

---

## 📜 License

This project is licensed under the MIT License.

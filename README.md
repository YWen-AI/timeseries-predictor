# Time Series Predictor

A Python project for training and predicting time series data using classical and machine learning models. Designed for experimentation, forecasting, and deployment-ready pipelines.

---

## 📦 Features

- Data preprocessing and feature engineering
- Multiple time series model support (ANN, ARIMA (WIP))
- Model training and evaluation
- Forecasting future values
- Visualizations for trends and predictions
- Configurable workflows (e.g., via YAML or CLI arguments)

---

## 📁 Project Structure

```
timeseries-predictor/
├── bike_sharing_timeseries_predictor/    # Core source code
│   ├── model/                 # Training model and predict 
│   └── utils/                 # Ploting
├── notebooks/                 # Jupyter notebooks for exploration
├── data/                      # Data folder
├── tests/                     # Unit tests
├── pyproject.toml             # Poetry project config
├── .gitignore
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/YWen-AI/timeseries-predictor.git
cd timeseries-predictor
```

### 2. Install dependencies

```bash
poetry install
```

### 3. Run training and prediction

```bash
poetry run python bike_sharing_timeseries_predictor/model/bike_rental_prediction_NN.py
```

---

## 🧪 Running Tests

```bash
poetry run pytest test/
```

---

## 📝 License

This project is licensed under the MIT License.

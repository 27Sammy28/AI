exoplanet_classifier/
├── data/
│   ├── raw/                  # Place original dataset CSV here
│   └── processed/            # Save cleaned & feature-engineered data
├── notebooks/
│   ├── exploration.ipynb     # Initial data exploration & charts
│   └── baseline_model.ipynb  # First ML models & testing
├── src/
│   ├── data_loader.py        # Load + basic clean functions
│   ├── features.py           # Feature engineering scripts
│   ├── train_model.py        # Training ML models
│   ├── evaluate.py           # Metrics, ROC, confusion matrix
├── models/                   # Save trained models (.pkl)
├── plots/                    # Charts, confusion matrix images
├── requirements.txt          # All dependencies
└── README.md                 # Project overview, dataset link, how to run

# src/data_loader.py
import pandas as pd

def load_raw_data(filepath):
    return pd.read_csv(filepath)

# src/features.py
def create_features(df):
    # Example: Drop missing, convert categorical
    df = df.dropna()
    return df

# src/train_model.py
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/model.pkl')
    return model

# src/evaluate.py
from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

# requirements.txt
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib

# README.md
# Exoplanet Habitability Classifier
This project trains machine learning models to classify potentially habitable exoplanets using physical and orbital features.

## Data Source
- [Planetary Habitability Laboratory](https://phl.upr.edu/projects/habitable-exoplanets-catalog/data)

## Run Instructions
1. Add CSV to `data/raw/`
2. Run notebook to explore
3. Use scripts in `src/` to process, train, and evaluate models


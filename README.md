## IIoT Predictive Maintenance — Maintenance Requirement Classifier

A lightweight Industrial Internet of Things (IIoT) project that trains and evaluates a model to predict whether equipment requires maintenance. The project includes data preprocessing, training with XGBoost, probability calibration, custom-threshold predictions, and small IoT demo utilities for message publishing/receiving.

---

## Contents
- `Train.py` — training pipeline: loads the dataset (`iiot_dataset.xlsx`), preprocesses data, trains an XGBoost classifier wrapped in `CalibratedClassifierCV`, saves the trained pipeline and test splits, and writes custom-threshold predictions.
- `Test.py` — evaluation / test script (see file for details).
- `X_test.csv`, `y_test.csv`, `y_pred_custom.csv` — outputs produced by `Train.py` when run.
- `iiot_dataset.xlsx` — expected dataset input (not included by default).
- `IOT/` — IoT demo utilities (publisher, receiver, styles): `fake_publisher.py`, `iot.py`, `rec.py`, `iot.css`.
- `result/` — folder for result outputs (`predictions.csv`, `test_results.txt`).
- `iot_messages.json` — example IoT messages (if present).

---

## Quick start (Windows PowerShell)

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install required packages (suggested minimal list):

```powershell
pip install pandas scikit-learn xgboost joblib openpyxl numpy
```

3. Place your dataset `iiot_dataset.xlsx` at the project root (or update the path inside `Train.py`) and run training:

```powershell
python .\Train.py
```

Expected artifacts after a successful run:

- `maintenance_model.pkl` — saved model pipeline (preprocessor + calibrated classifier)
- `X_test.csv`, `y_test.csv` — exported test split
- `y_pred_custom.csv` — predictions generated with the threshold defined in `Train.py`
- Console output including class-balance, classification report, and confusion matrix

---

## Requirements

Install dependencies from `requirements.txt` (recommended). From PowerShell:

```powershell
pip install -r requirements.txt
```

The repository includes a minimal `requirements.txt` listing the packages used by the project. Pin versions for reproducible environments if you plan to share or deploy.

---

## How `Train.py` works (summary)

- Loads `iiot_dataset.xlsx` and normalizes column names (strips spaces and special characters).
- Drops rows with missing values by default (`df.dropna()`).
- Applies a small manual feature weighting to emphasize energy and carbon (`Energy_Consumption_kWh` and `Carbon_Emission_kg` multiplied by 1.2).
- Preprocessing pipeline:
  - Numeric features scaled with `StandardScaler`: `Temperature_C`, `Vibration_Level_mms`, `Energy_Consumption_kWh`, `Carbon_Emission_kg`, `Downtime_Hours`
  - Categorical features one-hot encoded: `Industry`, `Region`
- Trains an `xgboost.XGBClassifier` and wraps it with `CalibratedClassifierCV(method='sigmoid')` for better probability estimates.
- Handles class imbalance by computing `scale_pos_weight = (neg_count / pos_count)` on the training labels.
- Uses a custom probability threshold (default 0.55 in the script) to convert probabilities to binary labels and saves them.

Note: Calibration and cross-validation add compute overhead — reduce `cv` or `n_estimators` for quick experiments.

---

## Quick inference example (load saved model and run on a new sample)

Below is a minimal example to load `maintenance_model.pkl` and run inference on a pandas DataFrame `df_sample`:

```python
import joblib
import pandas as pd

# load pipeline
clf = joblib.load('maintenance_model.pkl')

# df_sample should have the same columns used in training (Industry, Region, numeric features, etc.)
# example:
df_sample = pd.DataFrame([
    {
        'Temperature_C': 45.0,
        'Vibration_Level_mms': 1.2,
        'Energy_Consumption_kWh': 120.0,
        'Carbon_Emission_kg': 30.0,
        'Downtime_Hours': 2.0,
        'Industry': 'Manufacturing',
        'Region': 'Europe'
    }
])

# predict probabilities (class 1 = maintenance required)
proba = clf.predict_proba(df_sample)[:, 1]
threshold = 0.55
pred = (proba >= threshold).astype(int)
print('probability:', proba, 'prediction:', pred)
```

---

## IoT demo (simulated)

- Run the fake publisher to simulate sensor messages:

```powershell
python .\IOT\fake_publisher.py
```

- Run `iot.py` or `rec.py` to receive/record messages. These are demo scripts and may need small adjustments to match your environment (host/port, protocol).

---

## Common issues & troubleshooting

- Missing `iiot_dataset.xlsx`: place the file in the repo root or update the path in `Train.py`.
- Column mismatch: `Train.py` normalizes column names — if your dataset uses different feature names, update either the dataset or the script.
- All-negative or all-positive labels in training split: computing `scale_pos_weight` will fail if there are 0 positives; add a guard or resampling strategy.
- Long training times: lower `n_estimators`, reduce `cv` folds, or sample a smaller dataset during development.

---

## Suggested next steps

1. Add a `requirements.txt` or `environment.yml` for reproducible installs.
2. Add a small Jupyter notebook demonstrating model training, calibration curve, and threshold tuning.
3. Add a `Dockerfile` or GitHub Actions workflow for CI to run lint/tests and a quick smoke training.
4. Add unit tests (pytest) for data-loading, preprocessing, and the trained pipeline's input shape.

---

## License

This project is released under the MIT License — see the included `LICENSE` file for details.

---

---

## Contact

If you want help improving reproducibility (requirements file, CI, Dockerfile), or want me to add tests/Notebook/Dockerfile, reply here and I can add them.

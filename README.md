# Aiqon AI — Port Congestion Prediction & Shipment Delay Risk

A machine-learning prototype for predicting port congestion levels and identifying shipments at risk of delays of six hours or more.

> **Important:** This project uses a synthetic dataset. All reported performance metrics describe experiments on simulated data and do not establish real-world predictive performance.

## 1. Project Overview

The goal is to investigate whether operational conditions, weather, and port-capacity indicators can help predict congestion and shipment-delay risk.

The project includes:
- A multiclass congestion classifier: **Low, Moderate, or High**.
- A binary shipment-delay classifier: **delay of 6 hours or more**.
- A Streamlit dashboard for interactive predictions.
- Chronological training, validation, and held-out testing.
- Model comparisons, threshold selection, evaluation metrics, and test predictions.

## 2. Dataset

**File:** `data/realistic_port_congestion_800.csv`

- 800 synthetic records.
- Three ports: Mumbai, Mundra, and Visakhapatnam.
- Simulated dates from January 1 to September 23, 2024.
- Operational variables include vessel arrivals, berth capacity and utilization, backlog, previous waiting time, weather, labour and equipment availability, customs pressure, maintenance, and planned cargo.

The dataset was constructed after exploring available port-traffic, government, maritime, and weather data sources. Many accessible sources provided weekly or monthly aggregates rather than consistently time-aligned daily operational observations and next-day outcomes suitable for this prediction task.

**The dataset does not represent verified historical port operations.** Its patterns are intended to support a modelling prototype, not to establish statistical equivalence with real ports.

## 3. Targets and Leakage Prevention

The models predict two targets:

| Target | Prediction |
|---|---|
| `congestion_level` | Low / Moderate / High |
| `shipment_delayed_6h_or_more` | 0 = no simulated delay of at least 6 hours; 1 = delay of at least 6 hours |

The training pipeline excludes the following columns from model inputs:

- `observed_queue_length_vessels`
- `observed_waiting_time_hours`
- `observed_shipment_delay_hours`
- `congestion_level`
- `shipment_delayed_6h_or_more`
- Raw `date`, which is used to construct chronological splits

Calendar features already present in the dataset are retained. Numeric missing values are imputed using training-set medians; categorical missing values are imputed using the most frequent training-set value, followed by one-hot encoding. No resampling is used.

These exclusions help prevent the models from using outcome fields that would not be available as legitimate prediction-time inputs.

## 4. Modelling Approach

The project explores Logistic Regression, Random Forest, and a time-series modelling direction.

Logistic Regression was selected for the current model configuration based on validation results and the desired operational trade-off. For congestion, the decision threshold is adjusted to prioritize detecting High-congestion cases while limiting false alarms on validation data.

This is a prototype-level model choice, not a claim that Logistic Regression is universally superior or will perform better on real port data.

### Threshold selection

For High congestion, the threshold was selected on validation data to maximize recall while keeping the false-positive rate at or below 25%. The selected threshold is **0.24**.

In simple terms, the threshold was chosen to catch as many High-congestion cases as possible without exceeding the predefined validation false-alarm limit.

This constraint applies to the validation set; it does not guarantee the same false-positive rate on unseen or live data.

## 5. Evaluation Design

The dataset is split chronologically by date, without random shuffling:

| Split | Records | Period |
|---|---:|---|
| Training | 558 | January 1 – July 4, 2024 |
| Validation | 120 | July 5 – August 13, 2024 |
| Held-out test | 122 | August 14 – September 23, 2024 |

Validation data is used for model/configuration selection and threshold selection. The final test set is kept separate to estimate performance on later dates.

Because the dataset is small and class-imbalanced, accuracy is considered alongside balanced accuracy, macro-F1, class-level precision and recall, confusion matrices, and ROC-AUC where applicable.

## 6. Latest Held-Out Test Results

The following results are from the latest threshold-tuned model run, using the 122-record chronological test set.

### Congestion classification

| Metric | Result |
|---|---:|
| Selected model | Logistic Regression |
| Decision threshold for High congestion | 0.24 |
| Accuracy | 68.0% |
| Balanced accuracy | 72.8% |
| Macro-F1 | 63.0% |
| High-congestion recall | 92.9% |
| High-congestion precision | 35.1% |

The model detected **13 of the 14 actual High-congestion cases**, missing one. It generated 24 false High alerts among the 108 cases that were not High congestion. In total, 13 of the 37 High alerts were correct.

This reflects a recall-oriented operating point: most actual High cases were detected, but many alerts were false positives and require further investigation.

### Shipment-delay prediction

| Metric | Result |
|---|---:|
| Selected model | Logistic Regression |
| Decision threshold | 0.35 |
| Accuracy | 73.8% |
| Balanced accuracy | 78.3% |
| Macro-F1 | 66.8% |
| Delay recall | 85.0% |
| Delay precision | 37.0% |
| ROC-AUC | 87.6% |

The model detected 17 of the 20 delayed-shipment cases in the test set. Its relatively low precision means many predicted delay alerts were not actual positive cases in this simulated test set.

**Interpretation:** these metrics demonstrate the behaviour of the current prototype on synthetic data. They should not be interpreted as evidence of equivalent performance on real port operations.

### Earlier baseline comparison

The original baseline run is recorded separately in `reports/evaluation_metrics.json`. Its congestion macro-F1 and balanced accuracy were 0.562 and 0.605, respectively. The original delay model achieved 0.729 macro-F1 and 0.822 balanced accuracy.

These are earlier results, not the latest threshold-tuned results above. The delay model's latest configuration is not better than the original baseline on every metric, so model selection should be revisited when more representative data becomes available.

## 7. Dashboard

The Streamlit dashboard accepts operational inputs and displays:
- Predicted congestion class.
- Congestion class probabilities.
- Predicted shipment-delay risk.
- Model-generated delay probability.

The displayed probabilities are raw model outputs and are **not calibrated confidence scores**. The dashboard is a demonstration interface, not a production decision system.

## 8. Run Locally

### Requirements

- Python 3.10 or a compatible Python version supported by the listed dependencies.
- Git.
- The project files and dataset.

### Windows PowerShell

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/Srijavali/port_congestion_detection.git
cd port_congestion_detection
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train the models and generate evaluation artefacts:

```powershell
python .\src\train_models.py --data data/realistic_port_congestion_800.csv --out . --max-high-fpr 0.25 --beta 2 --delay-beta 2
```

Launch the dashboard:

```powershell
streamlit run .\app.py
```

The training script writes model artefacts to `models/` and evaluation outputs to `reports/`. Use the arguments shown above for the threshold-constrained training configuration.

## 9. Repository Structure

```text
port_congestion_detection/
├── app.py
├── README.md
├── requirements.txt
├── data/
│   └── realistic_port_congestion_800.csv
├── models/
│   ├── congestion__v2_selected.joblib
│   └── delay__v2_selected.joblib
├── reports/
│   ├── evaluation_metrics.json
│   ├── evaluation_metrics_v2.json
│   ├── data_quality_summary.json
│   └── test prediction files
└── src/
    └── train_models.py
```

Additional comparison models and report files may also be present in these directories.

## 10. Limitations and Next Steps

- The dataset and labels are synthetic, so external validity has not been established.
- Only three ports and fewer than nine months of simulated dates are represented.
- The test set contains only 14 High-congestion cases and 20 delayed-shipment cases, making performance estimates uncertain.
- False alerts remain substantial, and performance may differ across ports.
- Missing values are imputed; the assumptions and their effects should be assessed further.
- A real deployment would require timestamped operational data, documented label definitions, temporal backtesting across additional periods, probability calibration, drift monitoring, and validation with port and logistics stakeholders.

## 11. Project Repository

GitHub: https://github.com/Srijavali/port_congestion_detection

## Conclusion

This project demonstrates an end-to-end workflow for exploring port data, constructing a synthetic dataset, training and comparing models, applying leakage-aware chronological validation, and building an interactive prediction dashboard.

The current results are useful for understanding model behaviour and the trade-off between missed High-congestion cases and false alarms. Further evaluation using real, timestamped port data is necessary before drawing conclusions about operational usefulness.

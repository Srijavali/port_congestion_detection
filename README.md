# Aiqon AI — Port Congestion & Shipment Delay Risk (Synthetic Prototype)

## Important data note
This project uses `data/realistic_port_congestion_800.csv`, an 800-row **synthetic** dataset. It is not a record of actual port operations. Any measured performance describes this dataset only and must not be represented as validated real-world port prediction performance.

## What it predicts
1. `congestion_level`: Low / Moderate / High.
2. `shipment_delayed_6h_or_more`: binary target (1 means the simulated shipment delay is at least six hours).

The CSV also contains observed/simulated outcomes such as queue length, waiting time and shipment delay. These are evaluation targets or post-outcome fields, not input features.

## Leakage prevention
The training script excludes these columns from all model inputs:
- `observed_queue_length_vessels`
- `observed_waiting_time_hours`
- `observed_shipment_delay_hours`
- `congestion_level`
- `shipment_delayed_6h_or_more`
- raw `date` (used only to create chronological train/validation/test splits)

Calendar features already present in the file (day of week, month, weekend, holiday) are retained. Missing numeric inputs are imputed with training-set medians; categorical values are imputed with the most frequent training-set value and one-hot encoded. No resampling is used.

## Evaluation design
The data is split chronologically by unique date, not randomly shuffled:
- first 70% of dates: training
- next 15% of dates: validation period
- final 15% of dates: held-out test period

The current script reports accuracy, balanced accuracy, macro-F1, weighted-F1, confusion matrices, class-level precision/recall/F1 and ROC-AUC for the binary delay target. Because only 800 synthetic rows are available and class counts are imbalanced, per-class recall and macro-F1 matter more than accuracy alone. Validation results should guide model selection; test results should be reported once as final held-out performance.

Two models are compared: Logistic Regression and Random Forest. Do not assume Random Forest is better; select based on validation metrics and the operational cost of missed high-congestion/delayed cases.

## Run locally (Windows PowerShell)
```powershell
cd path\to\aiqon_port_congestion_project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python .\src\train_models.py
streamlit run .\app.py
```

Model artifacts are written to `models/`; metrics and held-out predictions are written to `reports/`.

## Dashboard
The Streamlit app takes prediction-time operational inputs and displays congestion class, the model's delay probability, and class probabilities. These are raw model probabilities, not calibrated confidence scores. The app is only a demonstration because the training data is synthetic.

## Limitations and next steps
- Synthetic labels and features cannot establish external validity.
- The dataset is imbalanced (Low congestion and no-delay records are more common).
- Weather/labour/equipment/customs fields contain missing values; the pipeline imputes them, but the assumptions should be disclosed.
- There are only three named ports and less than nine months of dates.
- A real deployment would require timestamped real-world inputs, documented label definitions, temporal backtesting across more periods, probability calibration, drift monitoring, and operational validation with port/logistics stakeholders.


## Baseline results from the current run

Using the chronological 70/15/15 split (558 train, 120 validation, 122 test rows), Logistic Regression scored congestion macro-F1 0.562 and balanced accuracy 0.605 on the held-out test set. Random Forest scored congestion macro-F1 0.439 and balanced accuracy 0.440. For the ≥6-hour delay target, Logistic Regression scored macro-F1 0.729, balanced accuracy 0.822 and ROC-AUC 0.873; delayed-class recall was 0.850 and precision 0.447. Random Forest had 0.836 accuracy but only 0.050 delayed-class recall, illustrating why accuracy is not enough for this imbalanced problem. These are results on synthetic data only. Full reports are in `reports/evaluation_metrics.json`.

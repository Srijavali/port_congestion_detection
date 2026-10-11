# Predicting Port Congestion and Shipment Delays Using AI

## 1. Objective
This prototype estimates a port's congestion category (Low, Moderate, High) and whether a shipment's simulated delay reaches six hours. The aim is to demonstrate a leakage-aware modelling pipeline, risk estimates, and an interactive interface.

## 2. Dataset and preparation
The experiment uses 800 rows of synthetic data covering Mumbai, Mundra and Visakhapatnam from 1 January to 23 September 2024. Candidate inputs include scheduled arrivals and vessel mix, planned cargo volume, berth capacity/utilisation, current backlog, prior-three-day mean waiting time, forecast weather, labour/equipment availability, customs pressure, maintenance and arrival variability. The file includes 800 records and 31 columns. It has no exact duplicate rows. Missing values occur in several weather and operational-availability columns and are handled inside the training pipeline with median/mode imputation.

The targets are `congestion_level` and `shipment_delayed_6h_or_more`. Class counts are 570 Low, 163 Moderate and 67 High congestion; 721 not delayed and 79 delayed. This imbalance makes accuracy alone insufficient.

**Synthetic-data disclosure:** These rows are simulated, not observed port records. Results evaluate behaviour on the supplied synthetic data only; they do not prove that the model predicts actual port congestion or shipment delays.

## 3. Methodology
Logistic Regression and Random Forest are compared for each classification target. Numeric features are median-imputed and standardised for Logistic Regression; categorical port names are mode-imputed and one-hot encoded. The Random Forest uses the same preprocessing pipeline. The target and all post-outcome columns (`observed_queue_length_vessels`, `observed_waiting_time_hours`, `observed_shipment_delay_hours`) are excluded from the features to prevent target leakage. Raw date is used only to construct a chronological split; supplied calendar fields remain as inputs.

Dates are sorted and split chronologically: the first 70% for training, the next 15% for validation, and the final 15% for held-out testing. This avoids training on later dates and testing on earlier ones. Metrics include accuracy, balanced accuracy, macro-F1, weighted-F1, class-level precision/recall/F1, confusion matrices and ROC-AUC for binary delay prediction.

## 4. Results
The chronological split contains 558 training rows (1 January–4 July), 120 validation rows (5 July–13 August), and 122 held-out test rows (14 August–23 September 2024). On validation data, Logistic Regression outperformed Random Forest on macro-F1 for both targets, so it is the more defensible current candidate; the test results are reported as a final check rather than used for tuning.

| Target | Model | Test accuracy | Balanced accuracy | Macro-F1 | Additional metric |
|---|---|---:|---:|---:|---|
| Congestion level | Logistic Regression | 0.615 | 0.605 | 0.562 | — |
| Congestion level | Random Forest | 0.672 | 0.440 | 0.439 | — |
| Delay ≥6 hours | Logistic Regression | 0.803 | 0.822 | 0.729 | ROC-AUC 0.873; delayed-class precision 0.447, recall 0.850 |
| Delay ≥6 hours | Random Forest | 0.836 | 0.520 | 0.500 | ROC-AUC 0.772; delayed-class precision 0.500, recall 0.050 |

For congestion, Logistic Regression has better balanced accuracy and macro-F1 even though Random Forest has higher raw accuracy. For delay detection, Logistic Regression detects 85% of delayed test cases at the cost of false positives (precision 0.447); Random Forest misses nearly all delayed cases at its default threshold despite higher raw accuracy. This illustrates why accuracy alone is misleading with imbalanced targets. The held-out test set contains 14 High-congestion examples and 20 delayed examples, so estimates remain uncertain and should be interpreted cautiously.

The dashboard displays raw class probabilities. They are not calibrated confidence scores, and no probability calibration has been performed.

## 5. Limitations and future work
The dataset is synthetic, small, class-imbalanced and limited to three ports and less than nine months. Missing operational variables are imputed. The target-generation mechanism may create patterns that do not match real ports. Next steps are to validate labels with port/logistics experts, obtain documented real data, perform longer temporal backtests, calibrate probabilities, monitor drift, and assess the cost of false negatives before any operational use.

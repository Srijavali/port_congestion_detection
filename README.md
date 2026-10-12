# Aiqon AI — Port Congestion Prediction & Shipment Delay Risk

A machine-learning prototype that estimates port congestion levels and predicts shipment-delay risk using operational indicators and weather forecasts.

> **Research prototype:** This project uses synthetic data. All reported metrics describe experiments on simulated records and do not establish predictive performance on real port operations.

## 1. Project Overview

Port congestion and shipment delays can arise from interacting factors such as vessel arrivals, berth capacity, existing backlog, waiting time, weather, labour and equipment availability, and cargo-handling constraints.

This project explores an end-to-end machine-learning workflow for:

- **Congestion classification:** Low, Moderate, or High.
- **Shipment-delay prediction:** Whether a simulated shipment delay reaches six hours or more.
- **Risk assessment:** Displaying class probabilities and threshold-based alerts.
- **Model evaluation:** Chronological training, validation, and held-out testing.
- **Interactive demonstration:** A Streamlit dashboard with operational inputs, weather forecasts, and prediction-specific sensitivity explanations.

The prototype is intended for research and decision-support experimentation, not autonomous operational decision-making.

## 2. Dataset and Data Exploration

**Dataset:** `data/realistic_port_congestion_800.csv`

| Property | Description |
|---|---|
| Records | 800 synthetic observations |
| Ports | Mumbai, Mundra, Visakhapatnam |
| Simulated period | 1 January–23 September 2024 |
| Original dataset features | 31 columns |
| Data type | Synthetic operational data and simulated targets |

### Data-source exploration

The project began by exploring PortWatch, Zenodo, port-traffic databases, government resources, and Indian port-related websites. The objective was to identify data suitable for daily, next-day congestion and shipment-delay prediction.

The available sources were fragmented. Some provided weekly or monthly aggregates, port-traffic information, or performance indicators without consistently aligned daily operational inputs and labelled outcomes.

Consequently, a synthetic dataset was constructed to support model development and end-to-end evaluation.

**Important:** The dataset does not represent verified historical operations at the three ports. Its patterns and labels are simulated and should not be interpreted as statistically representative of real port conditions.

### Operational features

The original dataset includes variables relating to:

- Vessel arrivals and vessel categories.
- Berth capacity and utilisation.
- Existing backlog and recent waiting time.
- Weather conditions and derived weather indicators.
- Labour and equipment availability.
- Customs-processing pressure and planned maintenance.
- Planned cargo volume and vessel characteristics.

These variables provide a structured basis for exploring how operational conditions may relate to congestion and shipment delays.

## 3. Target Definitions and Leakage Prevention

### Prediction targets

| Target | Meaning |
|---|---|
| `congestion_level` | Simulated congestion class: Low, Moderate, or High |
| `shipment_delayed_6h_or_more` | `0` = simulated delay below 6 hours; `1` = simulated delay of 6 hours or more |

The current CSV shows the following approximate waiting-time patterns across congestion labels:

| Congestion label | Observed waiting-time pattern |
|---|---|
| Low | Below 3 hours |
| Moderate | 3 to less than 8 hours |
| High | 8 hours or more |

These ranges describe the observed synthetic dataset. The original dataset-generation script was not included in the reviewed repository bundle, so the exact label-generation rules could not be independently verified from that script.

The thresholds are treated as project-specific assumptions informed by the exploration of port-related data sources—not as official, universal congestion standards. Queue length and waiting time are relevant operational indicators, but the current data does not establish fixed queue-length cutoffs as the rules used to generate these labels.

For example, a simulated waiting time of 2.5 hours falls within the observed Low band, 5 hours within the Moderate band, and 10 hours within the High band.

Shipment delay is a separate target: a simulated delay of 7 hours is positive (`1`), whereas a delay of 4 hours is negative (`0`).

### Leakage prevention

The training pipeline excludes outcome and post-outcome fields from model inputs, including:

- `observed_queue_length_vessels`
- `observed_waiting_time_hours`
- `observed_shipment_delay_hours`
- `congestion_level`
- `shipment_delayed_6h_or_more`
- Raw `date`, which is used for chronological splitting

Preprocessing is fitted within the training pipeline. Numeric missing values are imputed using training-set medians, while categorical values are imputed using the most frequent training-set value and then one-hot encoded.

These steps help prevent the model from using outcomes that would not legitimately be available at prediction time.

## 4. Modelling Approach

The project compares Logistic Regression with Random Forest and histogram-based gradient-boosting configurations. ARIMA/SARIMA-style time-series methods were also considered during exploration but were not retained as the final approach.

Logistic Regression was selected for the current model configuration based on validation results and the intended operating trade-off. The selected congestion configuration uses a no-calendar feature set with `C = 0.05`; the selected shipment-delay configuration uses a no-calendar feature set with `C = 0.2`.

The dashboard calculates calendar features for the prediction date, but the selected no-calendar models do not use calendar features as predictors.

### Congestion threshold selection

The High-congestion alert threshold is **0.24**. It was selected using validation data under a false-positive-rate cap of 25% and a minimum High-recall target of 75%, followed by selection of the highest-precision qualifying threshold.

At the selected validation operating point:

- High-congestion recall was 75% (18 of 24 High cases).
- The false-positive rate was 21.9% (21 false High alerts among 96 non-High records).

These constraints apply to the validation set. They do not guarantee the same performance on unseen or live data.

The shipment-delay model uses a threshold of **0.35**.

## 5. Evaluation Design

The dataset is split chronologically by date, without random shuffling.

| Split | Records | Period |
|---|---:|---|
| Training | 558 | 1 January–4 July 2024 |
| Validation | 120 | 5 July–13 August 2024 |
| Held-out test | 122 | 14 August–23 September 2024 |

Validation data is used for model/configuration selection and threshold tuning. The final test period is kept separate to estimate performance on later simulated dates.

Evaluation includes accuracy, balanced accuracy, macro-F1, class-level precision and recall, and ROC-AUC where applicable. Because the test set is small and class-imbalanced, no single metric is sufficient to assess the model.

## 6. Held-Out Test Results

The following results describe the current threshold-tuned configuration on the 122-record chronological test set.

### Congestion classification

| Metric | Result |
|---|---:|
| Model | Logistic Regression |
| High-congestion threshold | 0.24 |
| Accuracy | 68.0% |
| Balanced accuracy | 72.8% |
| Macro-F1 | 63.0% |
| High recall | 92.9% |
| High precision | 35.1% |

The model detected 13 of the 14 actual High-congestion cases and missed one. It generated 24 false High alerts among the 108 non-High cases.

In total, 13 of the 37 High alerts were correct. This is a recall-oriented operating point: it detects most High cases but produces a substantial false-alert burden.

### Shipment-delay prediction

| Metric | Result |
|---|---:|
| Model | Logistic Regression |
| Decision threshold | 0.35 |
| Accuracy | 73.8% |
| Balanced accuracy | 78.3% |
| Macro-F1 | 66.8% |
| Delay recall | 85.0% |
| Delay precision | 37.0% |
| ROC-AUC | 87.6% |

The model detected 17 of the 20 delayed-shipment cases in the test set. Its relatively low precision means many delay alerts were false positives on this simulated test set.

### Baseline comparison

The original baseline results are retained separately in `reports/evaluation_metrics.json`.

| Metric | Original congestion baseline | Threshold-tuned congestion | Original delay baseline | Selected delay model |
|---|---:|---:|---:|---:|
| Balanced accuracy | 60.5% | 72.8% | 82.2% | 78.3% |
| Macro-F1 | 56.2% | 63.0% | 72.9% | 66.8% |

The threshold-tuned congestion configuration improves the reported overall metrics relative to the original congestion baseline at the tested operating points. However, the comparison combines model/configuration choices with threshold changes and should not be interpreted as a pure model-only improvement.

The selected delay model is not better than the original baseline on balanced accuracy or macro-F1. Its recall-focused operating point should be considered alongside the stronger overall metrics from the original baseline.

## 7. Streamlit Dashboard

The project includes a Streamlit dashboard that provides an interactive view of the model outputs.

### Dashboard capabilities

- Port selection for Mumbai, Mundra, and Visakhapatnam.
- Automatic retrieval of weather and marine forecasts from Open-Meteo when available.
- Operational inputs such as vessel arrivals, berth capacity, backlog, waiting time, cargo, resource availability, and maintenance.
- Congestion-class predictions and class probabilities.
- Shipment-delay probability and threshold-based alerts.
- Grouped sensitivity explanations that compare current inputs with typical synthetic-dataset values.
- A technical view of individual-feature sensitivity.

### Prediction horizon and weather

The dashboard is configured for a **next-day assessment**. The prediction date is set automatically to tomorrow rather than selected manually.

The weather function filters forecast timestamps for the target calendar date. Forecast availability may be incomplete, so full-day coverage should be verified during testing. When the weather API fails, the app uses clearly labelled training-data median values for the demonstration rather than presenting those values as live weather.

The dashboard's probability outputs are raw model probabilities, not calibrated real-world confidence scores. Its sensitivity explanations describe model behaviour, not proven causal effects.

## 8. Run Locally

### Prerequisites

- Python 3.10 or a compatible Python version supported by the dependencies.
- Git.
- An internet connection for live weather retrieval.
- The project dataset, model files, and source code.

### Step 1: Clone the repository

Open Windows PowerShell:

```powershell
git clone https://github.com/Srijavali/port_congestion_detection.git
cd port_congestion_detection
```

### Step 2: Create a virtual environment

The project was run locally in a Python virtual environment (`venv`). Create one with:

```powershell
python -m venv .venv
```

### Step 3: Activate the virtual environment

For Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation because of its execution policy, you can use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

After activation, your terminal should display `(.venv)` at the beginning of the command prompt.

### Step 4: Install dependencies

With the virtual environment activated:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 5: Run the dashboard

The repository contains the selected model artifacts under `models/`. To launch the dashboard using those existing artifacts, run:

```powershell
streamlit run .\app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that address in your browser.

### Optional: Retrain the models

To rerun the training pipeline and regenerate model and evaluation artifacts:

```powershell
python .\src\train_models.py --data data/realistic_port_congestion_800.csv --out . --max-high-fpr 0.25 --beta 2 --delay-beta 2
```

Retraining may overwrite existing artifacts and produce results that differ from the currently reported run. Review the resulting metrics before replacing the saved models or updating the report.

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
│   ├── delay__v2_selected.joblib
│   ├── congestion_level__logistic_regression.joblib
│   ├── congestion_level__random_forest.joblib
│   ├── shipment_delayed_6h_or_more__logistic_regression.joblib
│   └── shipment_delayed_6h_or_more__random_forest.joblib
├── reports/
│   ├── evaluation_metrics.json
│   ├── evaluation_metrics_v2.json
│   ├── data_quality_summary.json
│   └── saved test-prediction files
└── src/
    └── train_models.py
```

The repository also contains technical-report material and may include additional comparison artifacts.

## 10. Limitations and Future Work

- **Synthetic data:** The dataset and labels are simulated; real-world validity has not been established.
- **Label provenance:** The original dataset-generation script was not included in the reviewed repository bundle. The exact source-code rules for assigning congestion labels could not be independently verified.
- **Small test sample:** The held-out test set contains only 14 High-congestion cases and 20 delayed-shipment cases, making the performance estimates uncertain.
- **False alerts:** The congestion model produces a substantial number of false High alerts, particularly for Visakhapatnam.
- **Uneven port performance:** Aggregate metrics do not fully capture the differences observed across ports.
- **Uncalibrated probabilities:** Probability calibration has not been performed, so the outputs should not be interpreted as empirically reliable confidence percentages.
- **Forecast coverage:** The dashboard must verify that sufficient weather observations are available for the intended forecast period.
- **Explanation limitations:** Sensitivity comparisons may involve correlated features and do not establish causation.
- **Future validation:** A real deployment would require timestamped operational records, documented ground-truth definitions, temporal backtesting across additional periods, probability calibration, drift monitoring, and evaluation with port and logistics stakeholders.

A separate experimental dataset with 57 columns was also explored. Its targets were generated using additional simulation rules, some of which use features supplied to the model. Its results are therefore not directly comparable with the original-label results and do not independently demonstrate improved real-world prediction.

## 11. Project Repository

**GitHub:** https://github.com/Srijavali/port_congestion_detection

## Conclusion

This project demonstrates an end-to-end machine-learning workflow covering data-source exploration, synthetic dataset construction, leakage-aware preprocessing, chronological validation, model comparison, threshold-based risk prediction, and an interactive Streamlit dashboard.

The results help illustrate the trade-off between detecting High-congestion cases and generating false alerts. Validation on real, timestamped port data is necessary before drawing conclusions about operational reliability or deploying the system in practice.

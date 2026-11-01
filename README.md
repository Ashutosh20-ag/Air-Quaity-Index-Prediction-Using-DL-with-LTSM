# Air Quality Forecasting with LSTM

Forecasting daily Air Quality Index (AQI) for the most polluted Indian cities using a Long Short-Term Memory (LSTM) deep learning model, with an interactive Streamlit dashboard.

> **Deep Learning Project** | BTech CSE, Semester V, Section B | Group ID: `BT_DL_13` | Project ID: `DL-013` | Mentor: Sachin Raghav

---

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Dataset](#dataset)
4. [Methodology](#methodology)
5. [Model Architecture](#model-architecture)
6. [Results](#results)
7. [Dashboard](#dashboard)
8. [Project Structure](#project-structure)
9. [Installation and Usage](#installation-and-usage)
10. [Retraining the Models](#retraining-the-models)
11. [Limitations](#limitations)
12. [Future Scope](#future-scope)
13. [Team](#team)
14. [Acknowledgements](#acknowledgements)

---

## Overview

Air pollution is a serious public health problem in India. Knowing the air quality in advance helps people take precautions and helps authorities plan actions (for example, GRAP in Delhi).

Daily AQI is a **time series**: today's value depends on the last few days and weeks. A plain regression model does not remember this sequence, whereas an **LSTM** can learn patterns across time through its forget, input and output gates.

This project:

- Trains an LSTM on 30 days of pollutant history (PM2.5, PM10, NO2, SO2, CO, O3, AQI) to predict the next day's AQI.
- Compares the LSTM against a naive baseline, Linear Regression and GRU, across different window sizes.
- Trains one model per city for the most polluted cities in the dataset.
- Serves everything in a Streamlit dashboard with a 7-day forecast, AQI categories, health advice, a city map and accuracy plots.

## Features

- **7-day AQI forecast** for each supported city, shown as colour-coded cards following the CPCB AQI categories
- **Health advice** based on the forecast category
- **Forecast chart** with the model's typical error (RMSE) as a shaded band
- **City comparison**: India map, ranking bar chart and multi-city forecast lines
- **Pollutant view**: recent PM2.5, PM10, NO2, SO2, CO, O3 levels and their correlation with AQI
- **Model accuracy tab**: backtest on the last 90 test days, error histogram, actual vs predicted scatter, and a table for all trained cities
- **CSV download** of the forecast
- Cities with poor model accuracy (R² below 0.5) are automatically hidden from the dashboard

## Dataset

- **Source:** [Air Quality Data in India (2015 - 2020)](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india) on Kaggle (CPCB data), file `city_day.csv`
- **Size:** 29,531 rows and 16 columns, 26 cities, daily readings
- **Features used:** `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `AQI`
- **Target:** next day's `AQI`

The dataset ends in July 2020, so forecasts shown in the dashboard are for the days right after the last record (July 2020), not live data.

## Methodology

1. **Data selection:** pick a city and keep the 7 features. For the multi-city models, the most polluted cities (by average AQI, with at least 1000 days of AQI data) are used.
2. **Missing values:** time-based interpolation, followed by backward and forward fill.
3. **Train/test split:** first 80% of days for training and last 20% for testing. The data is **not shuffled**, to respect time order.
4. **Scaling:** `MinMaxScaler` fitted **only on the training data**, then applied to the test data (prevents data leakage).
5. **Sliding window:** the previous 30 days of all 7 features are used as input to predict the next day's AQI. Input shape: `(samples, 30, 7)`.
6. **Training:** Adam optimizer, MSE loss, early stopping on validation loss.
7. **Evaluation:** RMSE, MAE and R² on the test set, compared with baselines.
8. **Forecasting:** recursive multi-step forecasting. Each predicted AQI is fed back into the window to predict the next day.

## Model Architecture

```
Input (30 days x 7 features)
   -> LSTM (32 units)
   -> Dropout (0.2)
   -> Dense (16, ReLU)
   -> Dense (1)  -> next-day AQI
```

| Setting | Value |
|---|---|
| Optimizer | Adam (learning rate 1e-3) |
| Loss | Mean Squared Error |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | patience 10, best weights restored |
| Validation split | 10% of training data |
| Window size | 30 days |

## Results

### Delhi: model comparison

Test set results (lower RMSE and MAE are better, higher R² is better):

| Model | RMSE | MAE | R² |
|---|---|---|---|
| GRU (window = 14) | 34.563 | 24.019 | 0.907 |
| GRU (window = 30) | 35.042 | 24.593 | 0.905 |
| LSTM (window = 30) | 35.158 | 24.554 | 0.904 |
| Linear Regression (window = 14) | 35.409 | 25.536 | 0.903 |
| Linear Regression (window = 30) | 35.608 | 25.480 | 0.901 |
| Linear Regression (window = 7) | 35.781 | 26.010 | 0.900 |
| LSTM (window = 7) | 36.956 | 26.005 | 0.894 |
| GRU (window = 7) | 37.315 | 26.847 | 0.892 |
| LSTM (window = 14) | 37.360 | 26.261 | 0.891 |
| Naive baseline (yesterday's AQI) | 44.969 | 31.463 | 0.843 |

The final LSTM (window = 30, retrained for the saved model) scored **RMSE 35.68, MAE 24.80, R² 0.901**.

**Key observations**

- Every learned model beats the naive baseline clearly (RMSE around 45 down to around 35, roughly 21% lower error).
- LSTM, GRU and Linear Regression perform **very close to each other**. The differences are small and can change slightly between training runs, so no single model should be called the clear winner on this dataset.
- Windows of 14 to 30 days work better than a 7-day window for the recurrent models.
- Predictions follow the trend and the major peaks well, but tend to lag the actual curve by about a day and under-predict sudden spikes.
- The error histogram is roughly bell-shaped and centred near zero, so the model is not systematically biased.

### Multi-city models

Separate LSTM models are trained for the most polluted cities: Ahmedabad, Amritsar, Delhi, Gurugram, Jaipur, Lucknow, Patna and Visakhapatnam. Per-city RMSE, MAE, R² and improvement over the naive baseline are stored in `models/summary.json` and shown in the dashboard's **Model accuracy** tab.

Accuracy varies by city. For example, the Patna model did not generalise well (negative R² on the test period), so cities below `MIN_R2 = 0.5` are hidden from the dashboard by default. This is a data-quality and distribution-shift issue, not something to hide: check the accuracy table before trusting any city's forecast.

## Dashboard

Run it locally (see [Installation and Usage](#installation-and-usage)). Tabs:

| Tab | What it shows |
|---|---|
| **Forecast** | Next 1 to 7 days as AQI-coloured cards, health advice, gauge of the last recorded AQI, history plus forecast chart, CSV download |
| **City comparison** | Map and ranking of cities by predicted AQI (tomorrow or week average), multi-city forecast lines |
| **Pollutants** | Latest pollutant values with 7-day change, trend plots, correlation with AQI |
| **Model accuracy** | Backtest plot, RMSE / MAE / R², error histogram, actual vs predicted scatter, all-city accuracy table |

<!-- Add your screenshots to docs/screenshots/ and uncomment the lines below.

![Forecast tab](docs/screenshots/forecast.png)
![City comparison](docs/screenshots/city_comparison.png)
![Model accuracy](docs/screenshots/model_accuracy.png)
-->

### AQI categories used (CPCB)

| AQI | Category |
|---|---|
| 0 - 50 | Good |
| 51 - 100 | Satisfactory |
| 101 - 200 | Moderate |
| 201 - 300 | Poor |
| 301 - 400 | Very Poor |
| 401 - 500 | Severe |

## Project Structure

```
.
├── app.py                      # Streamlit dashboard
├── air_quality_lstm.ipynb      # Notebook: EDA, training, baselines, experiments, multi-city training
├── data/
│   └── city_day.csv            # CPCB daily city AQI data (Kaggle)
├── models/
│   ├── <City>.keras            # Trained LSTM per city
│   ├── <City>_scaler.pkl       # Fitted MinMaxScaler per city
│   ├── <City>_recent.csv       # Last 120 days (input window for forecasting)
│   ├── <City>_backtest.csv     # Last 90 test days: actual vs predicted
│   └── summary.json            # Metrics for all trained cities
├── aqi_lstm_final.keras        # Final Delhi LSTM (window = 30)
├── aqi_lstm_model.keras        # First Delhi LSTM (64 + 32 units)
├── scaler.pkl                  # Scaler for the Delhi models
├── model_comparison.csv        # LSTM / GRU / Linear Regression / Naive comparison
└── .streamlit/config.toml      # Streamlit theme
```

## Installation and Usage

**Requirements:** Python 3.10 to 3.12 (developed with Python 3.11 and TensorFlow 2.21).

```bash
# 1. Clone the repository
git clone https://github.com/Ashutosh20-ag/Air-Quaity-Index-Prediction-Using-DL-with-LTSM.git
cd Air-Quaity-Index-Prediction-Using-DL-with-LTSM

# 2. Create and activate a virtual environment
python3.11 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn plotly streamlit joblib jupyter ipykernel

# 4. Run the dashboard
streamlit run app.py
```

The app opens in your browser (usually at `http://localhost:8501`). Stop it with `Ctrl + C`.

If you only want to run the dashboard, the trained models are already included in `models/`, so no training is needed.

## Retraining the Models

1. Make sure `data/city_day.csv` exists (download it from Kaggle if needed).
2. Open `air_quality_lstm.ipynb` in VS Code or Jupyter, select the virtual environment as the kernel, and run the cells in order.
3. The notebook covers EDA, preprocessing, the Delhi LSTM, baselines (naive and Linear Regression), the window and GRU experiments, and the multi-city training loop that regenerates everything in `models/`.
4. Training all cities takes around 8 to 12 minutes on a laptop CPU.

## Limitations

- **Not live data.** The dataset ends in July 2020, so forecasts are for July 2020. Real-time use would need a live feed (for example, CPCB or OpenAQ APIs).
- **Recursive forecast drift.** For the 7-day forecast, other pollutants are held at their last values, and each prediction is fed back as input. Forecasts therefore flatten towards the average and do not capture sudden spikes. Accuracy drops as the horizon grows.
- **Small dataset.** Roughly 2000 daily records per city limit what a deep model can learn, which is why a simple Linear Regression is competitive.
- **No weather features.** Temperature, wind speed, humidity and rainfall strongly affect pollution but are not in the dataset.
- **City-wise accuracy varies** (see the accuracy table), and some cities have long gaps in the data.
- Forecasts and health advice are for **educational purposes** and are not medical or official air quality advice.

## Future Scope

- Add weather features (temperature, wind, humidity, rainfall)
- Connect a real-time data source (CPCB / OpenAQ) for live forecasting
- Multi-step direct forecasting (predict all 7 days at once) instead of recursive forecasting
- Attention-based or Transformer models, and a single multi-city model
- Hyperparameter tuning and uncertainty estimation (for example, prediction intervals)
- Deploy the dashboard on Streamlit Community Cloud

## creator 

-Ashutosh20-ag
## Acknowledgements

- Dataset: Rohan Rao, *Air Quality Data in India (2015 - 2020)*, Kaggle (data originally from the Central Pollution Control Board, India)
- Built with TensorFlow / Keras, scikit-learn, pandas, Plotly and Streamlit

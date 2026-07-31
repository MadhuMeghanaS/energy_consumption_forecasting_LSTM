# Hourly Energy Consumption Forecasting with an LSTM (PyTorch)

An intermediate **time-series** project: predict the next hour's electricity
demand (megawatts) for the PJM Interconnection grid using a sliding-window
LSTM regressor.

## Dataset

**Kaggle: "Hourly Energy Consumption"**
Link: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

- Real hourly power consumption data (megawatts) from PJM, a regional
  transmission organization in the US, from 2002–2018.
- We use `PJME_hourly.csv` (PJM East region) — two columns: `Datetime`, `PJME_MW`.

### Download instructions
```bash
pip install kaggle
kaggle datasets download -d robikscube/hourly-energy-consumption -p data/ --unzip
```


## Folder structure
```
Energy_Pytorch/
├── data/                 
├── train.py               
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt
python train.py --data_dir data --epochs 15 --window 168
```

The script will:
1. Load and sort the time series, handle duplicate timestamps.
2. Min-max scale the series (fit on train split only, to avoid leakage).
3. Build sliding windows (X = past `window` hours, y = next hour).
4. Chronologically split into train/val/test (no shuffling — this is a time
   series, order matters!).
5. Train a stacked LSTM regressor with early stopping on validation loss.
6. Report test-set **MAE**, **RMSE**, and **MAPE** in the original MW scale.
7. Save the trained model to `model.pt` and a plot of predictions vs. actuals
   to `forecast_plot.png`.


## Future enhancements
- Add calendar features (hour-of-day, day-of-week, month, holiday flag) as
  extra input channels — this typically gives the biggest accuracy boost.
- Turn it into multi-step forecasting (predict the next 24 hours at once).
- Compare against a classical baseline (SARIMA or Facebook Prophet) and
  report the head-to-head numbers — a very interview-friendly narrative.
- Wrap the trained model in a small FastAPI endpoint that returns the next
  24-hour forecast given the latest 168 hours of data.

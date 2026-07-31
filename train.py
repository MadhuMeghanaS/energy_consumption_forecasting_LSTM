"""
Hourly Energy Consumption Forecasting with an LSTM (PyTorch)
Dataset: https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption
File used: PJME_hourly.csv

"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# Data utilities
def load_series(data_dir: str) -> pd.Series:
    df = pd.read_csv(os.path.join(data_dir, "PJME_hourly.csv"))
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.drop_duplicates(subset="Datetime").sort_values("Datetime")
    df = df.set_index("Datetime")
    # fill any small gaps in the hourly index by forward-filling
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="H")
    df = df.reindex(full_index).ffill()
    return df["PJME_MW"]


def make_windows(series: np.ndarray, window: int):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class MinMaxScalerManual:

    def fit(self, data):
        self.min_ = data.min()
        self.max_ = data.max()
        return self

    def transform(self, data):
        return (data - self.min_) / (self.max_ - self.min_ + 1e-8)

    def inverse_transform(self, data):
        return data * (self.max_ - self.min_) + self.min_


# Model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, 1)
        out, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (B, H)
        return self.fc(last_hidden).squeeze(1)


# Train / eval
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--window", type=int, default=168)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading series...")
    series = load_series(args.data_dir).values.astype(np.float32)
    print(f"Series length: {len(series)} hours")

    # Chronological split: 70% train, 15% val, 15% test
    n = len(series)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_raw = series[:train_end]
    val_raw = series[train_end - args.window : val_end]  # include lookback context
    test_raw = series[val_end - args.window :]

    scaler = MinMaxScalerManual().fit(train_raw)
    train_scaled = scaler.transform(train_raw)
    val_scaled = scaler.transform(val_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = make_windows(train_scaled, args.window)
    X_val, y_val = make_windows(val_scaled, args.window)
    X_test, y_test = make_windows(test_scaled, args.window)

    def to_loader(X, y, shuffle):
        X_t = torch.tensor(X).unsqueeze(-1)  # (N, T, 1)
        y_t = torch.tensor(y)
        return DataLoader(TensorDataset(X_t, y_t), batch_size=args.batch_size, shuffle=shuffle)

    train_loader = to_loader(X_train, y_train, shuffle=True)
    val_loader = to_loader(X_val, y_val, shuffle=False)
    test_loader = to_loader(X_test, y_test, shuffle=False)

    model = LSTMForecaster(hidden_size=args.hidden_size, num_layers=args.num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "model.pt")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    # --- Final test evaluation ---
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).unsqueeze(-1).to(device)
        preds_scaled = model(X_test_t).cpu().numpy()

    preds = scaler.inverse_transform(preds_scaled)
    actuals = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = np.mean(np.abs((actuals - preds) / actuals)) * 100

    print("\n=== Test set results (original MW scale) ===")
    print(f"MAE:  {mae:.2f} MW")
    print(f"RMSE: {rmse:.2f} MW")
    print(f"MAPE: {mape:.2f}%")

    # Plot last 500 hours of predictions vs actuals
    plt.figure(figsize=(14, 5))
    plt.plot(actuals[-500:], label="Actual", linewidth=1.5)
    plt.plot(preds[-500:], label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title("PJME Hourly Load: Predicted vs Actual (last 500 test hours)")
    plt.xlabel("Hour")
    plt.ylabel("Load (MW)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("forecast_plot.png", dpi=150)
    print("\nSaved best model to model.pt and plot to forecast_plot.png")


if __name__ == "__main__":
    main()

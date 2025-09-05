import numpy as np
import torch
import torch.nn as nn
import time
import xgboost as xgb
import matplotlib.pyplot as plt
import polars as pl
import json
from torch.utils.data import DataLoader, TensorDataset
from processing import get_data
from sklearn.metrics import root_mean_squared_error
from scipy.stats import norm
from typing import Literal
from pathlib import Path


# Self-Supervised LSTM Encoder
class LSTMEncoder(nn.Module):
    """
    Encodes a sequence window (T, d) into an embedding vector e_t.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1] # (batch, hidden_dim)


# Next-step prediction head
class NextStepHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int = 21):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        return self.fc(h)


# SSL Training Loop
def train_self_supervised(symbol: str, encoder: LSTMEncoder, head: NextStepHead, train_loader: DataLoader, epochs: int, lr: float, device: Literal["cpu", "cuda"] = "cpu") -> tuple[LSTMEncoder, nn.Sequential]:
    model = nn.Sequential(encoder, head).to(device)
    if Path("models", f"encoder_{symbol}.torch").is_file():
        encoder.load_state_dict(torch.load(f"models/encoder_{symbol}.torch", weights_only=True))
        model.load_state_dict(torch.load(f"models/model_{symbol}.torch", weights_only=True))
        return encoder, model

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n = 0
        i = 1
        total_len = len(train_loader)
        for xb, yb in train_loader:
            print(f"Batch: [{i} / {total_len}]")
            xb, yb = xb.to(device), yb.to(device)

            preds = model(xb)
            loss = loss_fn(preds.squeeze(), yb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
            n += xb.size(0)
            i += 1
        
        print(f"Epoch {epoch+1}, Loss = {total_loss / n:.6f}")

    return encoder, model # trained encoder, head discarded later


def extract_features(encoder: LSTMEncoder, X: np.ndarray, batch_size: int = 4096, device: Literal["cpu", "cuda"] = "cpu") -> np.ndarray:
    encoder.eval()
    feats = []

    loader = DataLoader(TensorDataset(torch.tensor(X.astype(np.float32))), batch_size=batch_size)
    with torch.no_grad():
        i = 1
        for xb in loader:
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb: torch.Tensor = xb.to(device)
            h = encoder(xb)
            feats.append(h.cpu())
            i += 1
    return torch.cat(feats).numpy()  # (n_samples, hidden_dim)


def model_predictions(model: nn.Sequential, X: np.ndarray, batch_size: int = 4096, device: Literal["cpu", "cuda"] = "cpu") -> np.ndarray:
    """Produce predictions for the non-XGBoost model (for comparison purposes)."""
    model.eval()
    preds = []

    loader = DataLoader(TensorDataset(torch.tensor(X.astype(np.float32))), batch_size=batch_size)
    with torch.no_grad():
        i = 1
        for xb in loader:
            print(f"Count: {i}")
            if isinstance(xb, (list, tuple)):
                xb = xb[0]
            xb: torch.Tensor = xb.to(device)
            h = model(xb)
            preds.append(h.cpu())
            i += 1
    return torch.cat(preds).numpy()


def produce_predictions(symbol: str, data: pl.DataFrame, save_models: bool, T: int, epochs: int, horizon: int = 21, train_fraction: float = 0.8, val_fraction: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]], list[tuple[float, float]], pl.Series, int]:
    """Produce predictions for the LSTM + XGBoost model."""
    assert train_fraction + val_fraction < 1
    # Filter data so there is only one entry per date
    proc_data = data.group_by("date").first().sort("date")

    # Process data
    X_raw = proc_data.drop("date").to_numpy()
    y_raw = proc_data["realized_volatility_21"].to_numpy()
    # Build self-supervised windows to predict next-step return
    X = np.array([X_raw[i : i + T] for i in range(len(proc_data) - T - horizon)], dtype=np.float32)
    y = np.array([y_raw[i + T : i + T + horizon] for i in range(len(proc_data) - T - horizon)], dtype=np.float32)
    date_col = proc_data["date"][:-T - horizon]


    # Dataset construction
    n = len(X)
    train_val_split = int(n * train_fraction)
    val_test_split = int(n * (train_fraction + val_fraction))
    X_train, X_val, X_test = X[:train_val_split], X[train_val_split:val_test_split], X[val_test_split:]
    y_train, y_val, y_test = y[:train_val_split], y[train_val_split:val_test_split], y[val_test_split:]

    dataset_train = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=False)

    # SSL Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 64
    encoder = LSTMEncoder(input_dim=X.shape[2], hidden_dim=hidden_dim, num_layers=1)
    head = NextStepHead(hidden_dim)
    trained_encoder, model = train_self_supervised(symbol, encoder, head, dataloader_train, epochs=epochs, lr=1e-3, device=device)
    
    # Extract embeddings
    train_embed = extract_features(trained_encoder, X_train, device=device)  # (split, emb_dim)
    val_embed = extract_features(trained_encoder, X_val, device=device)
    test_embed = extract_features(trained_encoder, X_test, device=device)

    # Train XGBoost on embeddings to predict volatility target
    dtrain = xgb.DMatrix(train_embed, label=y_train)
    dval = xgb.DMatrix(val_embed, label=y_val)
    dtest = xgb.DMatrix(test_embed, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist"
    }
    booster = xgb.train(params, dtrain, num_boost_round=200)
    if save_models:
        torch.save(trained_encoder.state_dict(), f"models/encoder_{symbol}.torch")
        torch.save(model.state_dict(), f"models/model_{symbol}.torch")
        booster.save_model(f"models/xgb_model_{symbol}.bin")

    # Uncertainty Quantification
    y_preds_val = booster.predict(dval)
    val_residuals = y_val - y_preds_val
    parameters = [norm.fit([elem[i] for elem in val_residuals]) for i in range(horizon)]
    conf_intervals = [norm.interval(0.95, *params) for params in parameters]

    y_preds_test = booster.predict(dtest)
    rmses = [root_mean_squared_error([elem[i] for elem in y_test], [elem[i] for elem in y_preds_test]) for i in range(horizon)]

    # print(f"[Downstream XGB] Vol target RMSE: {rmse:.6f}")
    for i, elem in enumerate(rmses):
        print(f"[XGB] {i}-Step RMSE: {elem:.6f}")
    print(f"Overall RMSE: {root_mean_squared_error(y_test, y_preds_test):.6f}")

    return y_test, y_preds_test, y_test - y_preds_test, conf_intervals, parameters, date_col, val_test_split


if __name__ == "__main__":
    start = time.perf_counter()

    symbol = "AMZN"
    data = get_data(symbol)
    T = 64
    num_epochs = 10
    y_test, y_preds, resid, conf_intervals, parameters, date_col, val_test_split = produce_predictions(symbol, data["date", "returns", "realized_volatility_5", "realized_volatility_11", "realized_volatility_21", "realized_volatility_60", "vol_of_vol_21"], save_models=True, T=T, epochs=num_epochs)

    print(f"Time taken: {time.perf_counter() - start} seconds")

    # Generate run configuration
    run_config = {"uncertainties": [float(elem[1]) for elem in parameters], "start_index": val_test_split}
    with open(f"models/run_config_{symbol}.json", "w") as file:
        json.dump(run_config, file)

    n = 10
    selected_preds = np.array([elem[n - 1] for elem in y_preds])
    plt.plot(date_col[val_test_split:], [elem[n - 1] for elem in y_test], label="Actual")
    plt.plot(date_col[val_test_split:], selected_preds, label=f"{n}-Step Predicted", c="red")
    plt.plot(date_col[val_test_split:], selected_preds + conf_intervals[n - 1][0], linestyle="dotted", alpha=0.5, c="red", label="95% Confidence Interval")
    plt.plot(date_col[val_test_split:], selected_preds + conf_intervals[n - 1][1], linestyle="dotted", alpha=0.5, c="red")
    plt.legend()
    plt.show()

    print("Done")
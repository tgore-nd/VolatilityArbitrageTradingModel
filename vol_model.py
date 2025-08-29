import numpy as np
import polars as pl
import duckdb
import torch
import torch.nn as nn
import time
import xgboost as xgb
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datetime import timedelta
from sklearn.metrics import root_mean_squared_error
from typing import Literal
from pathlib import Path


def create_sequences(data: np.ndarray, target: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate the targets for the model.

    Parameters
    ----------
    data : np.array of shape (N, F)   # features (returns, rv5, rv21, rv60, ...)
    target : np.array of shape (N,)   # next-day realized volatility
    window_size : int                 # lookback length
    
    Returns
    ----------
        X: np.array of shape (num_samples, window_size, F)
        y: np.array of shape (num_samples,)
    """
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])     # shape (window_size, F)
        y.append(target[i + window_size])     # scalar
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32) # X: (batch, time, features)


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
        return h_n[-1]  # (batch, hidden_dim)


# Next-step prediction head
class NextStepHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int = 1):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, h):
        return self.fc(h)


# SSL Training Loop
def train_self_supervised(encoder: LSTMEncoder, head: NextStepHead, train_loader: DataLoader, epochs: int = 5, lr: float = 1e-3, device: Literal["cpu", "cuda"] = "cpu", saved_model_path: str = "encoder.torch") -> tuple[LSTMEncoder, nn.Sequential]:
    if Path(saved_model_path).is_file():
        encoder.load_state_dict(torch.load(saved_model_path, weights_only=True))
        return encoder, nn.Sequential()

    model = nn.Sequential(encoder, head).to(device)
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

    return encoder, model  # trained encoder, head discarded later


def extract_features(encoder: LSTMEncoder, X: np.ndarray, batch_size: int = 128, device: Literal["cpu", "cuda"] = "cpu") -> np.ndarray:
    encoder.eval()
    feats = []

    loader = DataLoader(TensorDataset(torch.tensor(X.astype(np.float32))), batch_size=batch_size)
    with torch.no_grad():
        i = 1
        for xb in loader:
            print(f"Count: {i}")
            h = encoder(xb[0])
            feats.append(h.cpu())
            i += 1
    return torch.cat(feats).numpy()  # (n_samples, hidden_dim)


def get_data(symbol: str, equity_delta_path: str, options_delta_path: str, contract_id: str | None = None, filter_data_by_contract_duration: bool = True) -> pl.DataFrame:
    options_data = duckdb.sql(f"SELECT * FROM delta_scan('{options_delta_path}') WHERE symbol = '{symbol}' ORDER BY date").pl()
    equity_data = duckdb.sql(f"SELECT close, STRPTIME(timestamp, '%Y-%m-%d %H:%M:%S') AS date_col, date_trunc('day', date_col) AS date FROM delta_scan('{equity_delta_path}') WHERE ticker = '{symbol}' AND HOUR(date_col) = 16 AND MINUTE(date_col) = 0 ORDER BY date_col").pl().with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")).drop_nulls()
    data_full = options_data.join(equity_data["date", "close", "returns"], on="date", how="inner").with_columns(
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=5)) * np.sqrt(252)).alias("realized_volatility_5"),
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=21)) * np.sqrt(252)).alias("realized_volatility_21"),
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=60)) * np.sqrt(252)).alias("realized_volatility_60"),
        (pl.col("returns").pow(2)).alias("squared_returns")
    )

    if contract_id is None:
        chosen_contract: str = data_full["contractID"].sample(1).item()
    else:
        chosen_contract = contract_id
    
    if filter_data_by_contract_duration:
        data_full = data_full.filter(pl.col("contractID") == chosen_contract)
        return data_full["returns", "implied_volatility", "realized_volatility_5", "realized_volatility_21", "realized_volatility_60"]

    return data_full["returns", "realized_volatility_5", "realized_volatility_21", "realized_volatility_60"]


if __name__ == "__main__":
    start = time.perf_counter()

    # Acquire data
    symbol = "AAPL"
    equity_delta_path = r"C:\Users\tfgor\Documents\BayesianMarketMaker\data\deltalake" # remove explicit paths before posting this to GitHub
    options_delta_path = r"C:\Users\tfgor\Documents\OptionsVolatiltyTrader\data\deltalake"
    data = get_data(symbol, equity_delta_path, options_delta_path, filter_data_by_contract_duration=False)
    X_raw = data[:-1]
    y_raw = data["realized_volatility_21"].shift(-1).drop_nulls() # next day volatility

    # Build self-supervised windows to predict next-step return (no labels needed)
    T = 64
    X, y = create_sequences(X_raw.to_numpy(), y_raw.to_numpy(), T)  # self-supervised pairs (predict y_{t+T} from window ending at t + T - 1)


    # Chronological split
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    dataset_train = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    dataset_test = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=False)

    # SSL Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_dim = 64
    encoder = LSTMEncoder(input_dim=X.shape[2], hidden_dim=hidden_dim, num_layers=1)
    head = NextStepHead(hidden_dim)
    trained_encoder, model = train_self_supervised(encoder, head, dataloader_train, epochs=10, lr=1e-3, device=device)
    torch.save(trained_encoder.state_dict(), "encoder.torch")
    torch.save(model.state_dict(), "model.torch")

    # Extract embeddings
    train_embed = extract_features(trained_encoder, X_train, device=device)  # (split, emb_dim)
    test_embed = extract_features(trained_encoder, X_test, device=device)

    # Train XGBoost on embeddings to predict volatility target
    dtrain = xgb.DMatrix(train_embed, label=y_train)
    dtest = xgb.DMatrix(test_embed, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist"  # fast
    }

    booster = xgb.train(params, dtrain, num_boost_round=200)

    y_preds = booster.predict(dtest)
    rmse = root_mean_squared_error(y_test, y_preds)
    print(f"[Downstream XGB] Vol target RMSE: {rmse:.6f}")
    print(f"Time taken: {time.perf_counter() - start} seconds")

    plt.plot(y_test, label="Actual")
    plt.plot(y_preds, label="Predicted")
    plt.show()
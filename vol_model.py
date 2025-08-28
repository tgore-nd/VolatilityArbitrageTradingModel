# ------------------------------------------------------------
# Phase 1: Train an LSTM encoder by predicting next-step returns (self-supervised).
# Phase 2: Freeze encoder, extract embeddings, train XGBoost to predict volatility.
# ------------------------------------------------------------

import numpy as np
import polars as pl
import duckdb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import timedelta


def sliding_windows(series: np.ndarray, T: int, horizon: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (window, next-step) pairs from a 1D series.
    X[i] = series[i : i + T]
    y[i] = series[i + T : i + T + horizon]
    """
    X = []
    y = []
    N = len(series)
    for i in range(N - T - horizon + 1):
        X.append(series[i : i + T])
        y.append(series[i + T : i + T + horizon])
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    return X, y

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # X: (n_samples, T) or (n_samples, T, d)
        # y: (n_samples, horizon)
        if X.ndim == 2:
            X = X[..., None]  # make (n, T, 1)
        if y.ndim == 1:
            y = y[..., None]  # make (n, 1)
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Self-Supervised LSTM Encoder
class LSTMEncoder(nn.Module):
    """
    Encodes a sequence window (T, d) into an embedding vector e_t.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.bidirectional = bidirectional
        self.out_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor):
        # x: (batch, T, d)
        out, (h, c) = self.lstm(x)            # h: (num_layers*(2 if bi else 1), batch, hidden)
        last = h[-1] if not self.bidirectional else torch.cat([h[-2], h[-1]], dim=-1)
        # last: (batch, out_dim)
        return last

class NextStepHead(nn.Module):
    """
    Predicts the next step(s) (horizon) of the input series from the embedding.
    """
    def __init__(self, emb_dim: int, horizon: int = 1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Linear(emb_dim, horizon)
        )

    def forward(self, e):
        return self.fc(e)

# ==============  Training loop (self-supervised)  ==============

def train_self_supervised(encoder: LSTMEncoder, head, loader, epochs=10, lr=1e-3, device="cpu"):
    encoder.to(device)
    head.to(device)
    params = list(encoder.parameters()) + list(head.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    encoder.train()
    head.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for xb, yb in loader:
            # xb: (batch, T, d); yb: (batch, horizon)
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)  # shape (batch, horizon)
            emb = encoder(xb)
            pred = head(emb)                # (batch, horizon)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
            n += xb.size(0)
        print(f"[SelfSup] Epoch {ep:02d} | MSE: {total / n:.6f}")

    # freeze
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()
    head.eval()

# ==============  Embedding extraction  ==============

def extract_embeddings(encoder, X: np.ndarray, batch_size: int = 256, device="cpu") -> np.ndarray:
    if X.ndim == 2:
        X = X[..., None]
    ds = TensorOnlyDataset(torch.from_numpy(X.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    embs = []
    encoder.to(device)
    encoder.eval()
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device)
            e = encoder(xb).cpu().numpy()
            embs.append(e)
    return np.vstack(embs)

class TensorOnlyDataset(Dataset):
    def __init__(self, X: torch.Tensor):
        self.X = X
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i]


def get_data(symbol: str, equity_delta_path: str, options_delta_path: str, contract_id: str | None = None, filter_data_by_contract_duration: bool = True) -> pl.DataFrame:
    options_data = duckdb.sql(f"SELECT * FROM delta_scan('{options_delta_path}') WHERE symbol = '{symbol}' ORDER BY date").pl()
    equity_data = duckdb.sql(f"SELECT close, STRPTIME(timestamp, '%Y-%m-%d %H:%M:%S') AS date_col, date_trunc('day', date_col) AS date FROM delta_scan('{equity_delta_path}') WHERE ticker = '{symbol}' AND HOUR(date_col) = 16 AND MINUTE(date_col) = 0 ORDER BY date_col").pl().with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")).drop_nulls()
    data_full = options_data.join(equity_data["date", "close", "returns"], on="date", how="inner").with_columns(
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=5)) * np.sqrt(252)).alias("realized_volatility_5"),
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=21)) * np.sqrt(252)).alias("realized_volatility_21"),
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=60)) * np.sqrt(252)).alias("realized_volatility_60"),
        (pl.col("returns").pow(2)).alias("squared_returns"))

    if contract_id is None:
        chosen_contract: str = data_full["contractID"].sample(1).item()
    else:
        chosen_contract = contract_id
    
    if filter_data_by_contract_duration:
        data_full = data_full.filter(pl.col("contractID") == chosen_contract)
        return data_full["date", "returns", "implied_volatility", "realized_volatility_5", "realized_volatility_21", "realized_volatility_60"]

    return data_full["date", "returns", "realized_volatility_5", "realized_volatility_21", "realized_volatility_60"]

# ==============  Demo / Template main  ==============

if __name__ == "__main__":
    # Acquire data
    symbol = "AAPL"
    equity_delta_path = r"C:\Users\tfgor\Documents\BayesianMarketMaker\data\deltalake" # remove explicit paths before posting this to GitHub
    options_delta_path = r"C:\Users\tfgor\Documents\OptionsVolatiltyTrader\data\deltalake"
    X = get_data(symbol, equity_delta_path, options_delta_path, filter_data_by_contract_duration=False)[1:]
    y = X["realized_volatility_21"].shift(-1).drop_nulls() # next day volatility

    # --------- (1) Prepare data  ---------
    # rng = np.random.default_rng(7)

    # # Synthetic 1D "returns" with volatility clustering (ARCH-like)
    # N = 6000
    # eps = rng.standard_normal(N).astype(np.float32)
    # vol = np.zeros(N, dtype=np.float32)
    # ret = np.zeros(N, dtype=np.float32)
    # vol[0] = 0.2
    # for t in range(1, N):
    #     vol[t] = 0.05 + 0.9 * vol[t-1]**2 + 0.1 * eps[t-1]**2
    #     ret[t] = np.sqrt(abs(vol[t])) * eps[t]

    # Generate targets
    target_horizon = 1
    y = 

    # Suppose your *supervised* target is next-day realized variance (toy proxy)
    # You would replace this with your true vol target (e.g., realized variance from intraday data).
    target_horizon = 1
    y = (ret**2).astype(np.float32)  # toy: next-step variance proxy

    # Build self-supervised windows to predict next-step return (no labels needed)
    T = 64
    horizon = 1  # predict next 1 step of the *input* series
    X_ss, y_ss = sliding_windows(ret, T=T, horizon=horizon)  # self-supervised pairs
    # Align supervised targets to the same index (predict y_{t+T} from window ending at t + T - 1)
    y_sup = y[T:T + len(X_ss)]  # shape matches X_ss count

    # Chronological split (no shuffling!)
    n = len(X_ss)
    split = int(n * 0.8)
    X_ss_tr, X_ss_te = X_ss[:split], X_ss[split:]
    y_ss_tr, y_ss_te = y_ss[:split], y_ss[split:]
    y_sup_tr, y_sup_te = y_sup[:split], y_sup[split:]

    ds_tr = WindowDataset(X_ss_tr, y_ss_tr)
    ds_te = WindowDataset(X_ss_te, y_ss_te)
    dl_tr = DataLoader(ds_tr, batch_size=128, shuffle=False)  # keep chronological order
    dl_te = DataLoader(ds_te, batch_size=128, shuffle=False)

    # --------- (2) Self-supervised train  ---------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = LSTMEncoder(input_dim=1, hidden_dim=32, num_layers=1, bidirectional=False)
    head = NextStepHead(emb_dim=encoder.out_dim, horizon=horizon)

    train_self_supervised(encoder, head, dl_tr, epochs=10, lr=1e-3, device=device)

    # (Optional) quick self-supervised validation loss
    with torch.no_grad():
        loss_fn = nn.MSELoss()
        tot, count = 0.0, 0
        for xb, yb in dl_te:
            xb = xb.to(device)
            yb = yb.to(device).squeeze(-1)
            pred = head(encoder(xb))
            loss = loss_fn(pred, yb)
            tot += loss.item() * xb.size(0)
            count += xb.size(0)
        print(f"[SelfSup] Val MSE: {tot / count:.6f}")

    # --------- (3) Freeze encoder & extract embeddings for ALL windows ---------
    E_tr = extract_embeddings(encoder, X_ss_tr, device=device)  # (split, emb_dim)
    E_te = extract_embeddings(encoder, X_ss_te, device=device)  # (n-split, emb_dim)

    # Optionally concatenate exogenous/tabular features here:
    # Z_tr = np.concatenate([E_tr, exog_tr], axis=1)
    # Z_te = np.concatenate([E_te, exog_te], axis=1)
    Z_tr, Z_te = E_tr, E_te

    # --------- (4) Train XGBoost on embeddings to predict volatility target ---------
    import xgboost as xgb
    from sklearn.metrics import mean_squared_error

    dtrain = xgb.DMatrix(Z_tr, label=y_sup_tr)
    dtest  = xgb.DMatrix(Z_te, label=y_sup_te)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist"  # fast
    }

    bst = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dtest, "valid")],
        early_stopping_rounds=50,
        verbose_eval=50
    )

    yhat = bst.predict(dtest)
    rmse = mean_squared_error(y_sup_te, yhat, squared=False)
    print(f"[Downstream XGB] Vol target RMSE: {rmse:.6f}")

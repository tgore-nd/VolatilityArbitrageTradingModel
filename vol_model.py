import numpy as np
import torch
import torch.nn as nn
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# LSTM model
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        out, (h, c) = self.lstm(x)
        emb = h[-1] # last layer hidden state
        y_pred = self.fc(emb).squeeze(-1)
        return y_pred, emb
    

# Training loop
def train_lstm(data: DataLoader, num_epochs: int):
    input_dim = 1
    hidden_dim = 32
    model = LSTMForecaster(input_dim, hidden_dim)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        for xb, yb in data: # iterate through batches
            y_pred, _ = model(xb)
            loss = loss_fn(y_pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch + 1}, loss {loss.item():.4f}") # pyright: ignore[reportPossiblyUnboundVariable]
    
    return model

# Hidden layers of LSTM
def extract_embeddings(data: DataLoader, trained_model: LSTMForecaster):
    trained_model.eval()
    embeddings = []
    targets = []
    with torch.no_grad():
        for xb, yb in data:
            _, emb = trained_model(xb)
            embeddings.append(emb)
            targets.append(yb)
    
    return torch.cat(embeddings).numpy(), torch.cat(targets).numpy()

# 
def train_xgb(embeddings: np.ndarray, targets: np.ndarray, exog: np.ndarray) -> xgb.Booster:
    X_train, X_val, y_train, y_val = train_test_split(embeddings, targets, test_size=0.2, shuffle=False)

    train_data = xgb.DMatrix(X_train, label=y_train)
    val_data = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "reg:squarederror",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }

    evallist = [(train_data, 'train'), (val_data, 'eval')]
    bst = xgb.train(params, train_data, num_boost_round=300, evals=evallist,
                    early_stopping_rounds=20, verbose_eval=50)
    
    return bst


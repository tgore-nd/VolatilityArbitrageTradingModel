import polars as pl
import numpy as np
import torch
import xgboost as xgb
import matplotlib.pyplot as plt
from vol_model import LSTMEncoder, produce_predictions, extract_features
from processing import get_data
from dateutil.relativedelta import relativedelta

def rescale_iv(iv: float, T: int, h: int) -> float:
    return iv * np.sqrt(T / h)

def load_trained_models(encoder_path: str, xgb_path: str, num_features: int = 6, hidden_dim: int = 64, num_layers: int = 1) -> tuple[LSTMEncoder, xgb.Booster]:
    # Instantiate models
    booster = xgb.Booster()
    encoder = LSTMEncoder(input_dim=num_features, hidden_dim=hidden_dim, num_layers=num_layers)

    # Load saved models
    encoder.load_state_dict(torch.load(encoder_path, weights_only=True))
    booster.load_model(xgb_path)

    return encoder, booster

def generate_trades(test_data: pl.DataFrame, fit_uncertainties: list[float], max_horizon: int = 21, window_size: int = 64, z_threshold: float = 1.93) -> tuple[pl.Series, list[float]]:
    X_raw_orig = test_data.group_by("date").first().sort("date")[2853:]
    date_series = X_raw_orig["date"]
    X_raw = X_raw_orig["returns", "realized_volatility_5", "realized_volatility_11", "realized_volatility_21", "realized_volatility_60", "vol_of_vol_21"].to_numpy()
    X = np.array([X_raw[i : i + window_size] for i in range(len(X_raw) - window_size - max_horizon)], dtype=np.float32)
    encoder, booster = load_trained_models("encoder.torch", "xgb_model.bin")
    trades = {
        "puts_long": [], # (contract_id, rand_id, strike_price, premium_when_position_opened, close_position_date, exp_date, enter_date)
        "puts_short": [],
        "num_underlying": 0,
        "total_profit": 0,
    }
    total_profit = []
    trades_conducted = 0
    for i, current_date in enumerate(date_series[:len(X)]):
        print(f"[Iteration {i}] \t Date: {current_date} \t Profit: {trades['total_profit']} \t Num underlying: {trades['num_underlying']}")
        # Get data for the right date
        current_feature = X[i]
        current_date_data = test_data.filter(pl.col("date") == current_date)

        # Select the most ATM put
        S0 = current_date_data["close"].unique().item()
        current_date_data = current_date_data.with_columns(-(pl.col("strike") - S0).abs().alias("__atm_dist"))
        atm_put = current_date_data.sort("__atm_dist", "volume", descending=True).filter(pl.col("type") == "put").limit(1)

        # Close positions due to be closed
        for j, position in enumerate(trades["puts_long"]):
            if position[4] <= current_date and position[1]: # close the position
                option_data = current_date_data.filter(pl.col("contractID") == position[0])
                if option_data.is_empty(): option_data = test_data.filter(pl.col("contractID") == position[0])[-1]
                trade_price = np.random.uniform(option_data["bid"].item(), option_data["ask"].item())
                trades["total_profit"] += trade_price * 100
                trades["puts_long"][j][1] = False
                trades_conducted += 1
        for j, position in enumerate(trades["puts_short"]):
            if position[4] <= current_date and position[1]: # close the position
                option_data = current_date_data.filter(pl.col("contractID") == position[0])
                if option_data.is_empty(): option_data = test_data.filter(pl.col("contractID") == position[0])[-1]
                trade_price = np.random.uniform(option_data["bid"].item(), option_data["ask"].item())
                trades["total_profit"] -= trade_price * 100
                trades["puts_short"][j][1] = False
                trades_conducted += 1

        # Get predictions
        embeddings = extract_features(encoder, current_feature)
        preds = booster.predict(xgb.DMatrix(np.array([embeddings]))).ravel()

        # Get DTE and (raw) IV
        T_put = atm_put["expiration"].item() - current_date
        put_raw_iv = atm_put["implied_volatility"].item()
        
        # Decide which volatility to short
        # Compute z-scores
        put_z_scores = np.array([(rescale_iv(put_raw_iv, T_put.days, h + 1) - preds[h]) / fit_uncertainties[h] for h in range(max_horizon)])[:T_put.days]

        # Execute trades (close position when z changes sign)
        if T_put.days != 0:
            next_z_score = put_z_scores[0]
            trade_price = np.random.uniform(atm_put["bid"].item(), atm_put["ask"].item())
            if abs(next_z_score) > z_threshold: # only act when we are ~95% confident that there is a signal
                if next_z_score > 0:
                    # Sell short (overpriced IV)
                    # Find position close date
                    i = int(np.argmax(put_z_scores < 0)) if np.argmax(put_z_scores < 0) != 0 else None
                    if i is None:
                        close_date = atm_put["expiration"].item()
                    else:
                        close_date = current_date + relativedelta(days = i + 1)

                    trades["puts_short"].append([atm_put["contractID"].item(), True, atm_put["strike"].item(), trade_price, close_date, atm_put["expiration"].item(), current_date])
                    trades["total_profit"] += trade_price * 100
                    trades_conducted += 1
                elif next_z_score < 0:
                    # Long (underpriced IV)
                    i = int(np.argmax(put_z_scores > 0)) if np.argmax(put_z_scores > 0) != 0 else None
                    if i is None:
                        close_date = atm_put["expiration"].item()
                    else:
                        close_date = current_date + relativedelta(days = i + 1)

                    trades["puts_long"].append([atm_put["contractID"].item(), True, atm_put["strike"].item(), trade_price, close_date, atm_put["expiration"].item(), current_date])
                    trades["total_profit"] -= trade_price * 100
                    trades_conducted += 1
            
        # Delta hedge
        num_stocks = 0
        for position in trades["puts_long"]:
            if position[1]:
                option_data = current_date_data.filter(pl.col("contractID") == position[0])
                if option_data.is_empty(): option_data = test_data.filter(pl.col("contractID") == position[0])[-1]
                num_stocks += int(round(option_data["delta"].item()))
        for position in trades["puts_short"]:
            if position[1]:
                option_data = current_date_data.filter(pl.col("contractID") == position[0])
                if option_data.is_empty(): option_data = test_data.filter(pl.col("contractID") == position[0])[-1]
                num_stocks -= int(round(option_data["delta"].item()))
        trades["total_profit"] -= (num_stocks - trades["num_underlying"]) * S0
        trades["num_underlying"] += num_stocks - trades["num_underlying"]
        total_profit.append(trades["total_profit"])

    print(f"Final profit: {trades['total_profit']}")
    print(f"Num underlying: {trades['num_underlying']}")
    print(f"Total num trades: {trades_conducted}")

    return date_series[:len(X)], total_profit


if __name__ == "__main__":
    data = get_data("AAPL")
    uncertainties = [0.0397472158074379, 0.04684538394212723, 0.05283116176724434, 0.057300008833408356, 0.06144201382994652, 0.06536895781755447, 0.06939872354269028, 0.07407503575086594, 0.07858895510435104, 0.08284837007522583, 0.08689523488283157, 0.08914045989513397, 0.08884916454553604, 0.09038390964269638, 0.09106902778148651, 0.09349182993173599, 0.09508904814720154, 0.09517481178045273, 0.09482354670763016, 0.09399273246526718, 0.09394287317991257]
    date_series, total_profit = generate_trades(data, uncertainties)

    plt.plot(date_series, total_profit)
    plt.title("AAPL Volatility Arbitrage")
    plt.xlabel("Time")
    plt.ylabel("Profit")
    plt.show()
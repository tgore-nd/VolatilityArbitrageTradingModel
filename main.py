import duckdb
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta

if __name__ == "__main__":
    stockdata_path = r"C:\Users\tfgor\Documents\BayesianMarketMaker\data\deltalake"
    symbol = "AAPL"
    data = duckdb.sql(f"SELECT * FROM delta_scan('data\deltalake') WHERE symbol = '{symbol}' ORDER BY date").pl()
    stockdata = duckdb.sql(f"SELECT close, STRPTIME(timestamp, '%Y-%m-%d %H:%M:%S') AS date_col, date_trunc('day', date_col) AS date FROM delta_scan('{stockdata_path}') WHERE ticker = '{symbol}' AND HOUR(date_col) = 16 AND MINUTE(date_col) = 0 ORDER BY date_col").pl()
    stockdata = stockdata.with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")).drop_nulls()
    data_full = data.join(stockdata["date", "close", "returns"], on="date", how="inner")
    
    data_full = data_full.with_columns((pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=21)) * np.sqrt(252)).alias("act_vol"))
    chosen_contract = data_full["contractID"].sample(1)
    data_full = data_full.filter(pl.col("contractID") == chosen_contract)
    plt.plot(data_full["date"], data_full["implied_volatility"], label="Implied Volatility")
    plt.plot(data_full["date"], data_full["act_vol"], label="Annualized Actual Volatility")
    plt.title(f"Contract: {chosen_contract.item()}")
    plt.legend()
    plt.show()
    print("Done")

import duckdb
import polars as pl
import numpy as np
from datetime import timedelta


def get_data(symbol: str, equity_delta_path: str, options_delta_path: str, contract_id: str | None = None, filter_data_by_contract_duration: bool = False, only_model_data: bool = False) -> pl.DataFrame:
    """
    Acquire the dataset for model fitting. Note that the data itself is not included, as my broker would (obviously) not like that.

    Parameters
    ----------
    symbol : str
        The stock symbol for which to get data.
    equity_delta_path : str
        The path to the delta lake containing minute OHLCV equity data. This is aggregated into daily data.
    options_delta_path : str
        The path to the delta lake containing daily options data.
    contract_id : str | None
        The options contract ID used to filter the data. If `None` (default), a random option is chosen.
    filter_data_by_contract_duration : bool
        Whether or not to filter the data to include only intervals when option `contract_id` is active. This is `False` by default since we usually don't filter data. It can be nice for visualization purposes.
    only_model_data : bool
        Whether or not to filter the data to include only features used by the model. Default is `False`.
    
    Returns
    -------
    data : pl.DataFrame
        A full Polars DataFrame of all option, equity, and computed volatility data needed to fit the model.
    """
    options_data = duckdb.sql(f"SELECT * FROM delta_scan('{options_delta_path}') WHERE symbol = '{symbol}' ORDER BY date").pl()
    equity_data = duckdb.sql(f"SELECT close, STRPTIME(timestamp, '%Y-%m-%d %H:%M:%S') AS date_col, date_trunc('day', date_col) AS date FROM delta_scan('{equity_delta_path}') WHERE ticker = '{symbol}' AND HOUR(date_col) = 16 AND MINUTE(date_col) = 0 ORDER BY date_col").pl().with_columns((pl.col("close") / pl.col("close").shift(1) - 1).alias("returns")).drop_nulls()
    data_full = options_data.join(equity_data["date", "close", "returns"], on="date", how="inner").with_columns(
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=5)) * np.sqrt(252)).alias("realized_volatility_5"),
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=11)) * np.sqrt(252)).alias("realized_volatility_11"),
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=21)) * np.sqrt(252)).alias("realized_volatility_21"),
        (pl.col("returns").rolling_std_by(pl.col("date"), timedelta(days=60)) * np.sqrt(252)).alias("realized_volatility_60"),
        (pl.col("returns").pow(2)).alias("squared_returns")
    ).with_columns(
        (pl.col("realized_volatility_21").rolling_std_by(pl.col("date"), timedelta(days=21)) * np.sqrt(252)).alias("vol_of_vol_21")
    )

    if contract_id is None:
        chosen_contract: str = data_full["contractID"].sample(1).item()
    else:
        chosen_contract = contract_id
    
    if filter_data_by_contract_duration:
        data_full = data_full.filter(pl.col("contractID") == chosen_contract)
        if only_model_data:
            return data_full["date", "returns", "implied_volatility", "realized_volatility_5", "realized_volatility_11", "realized_volatility_21", "realized_volatility_60"]
        return data_full

    if only_model_data:
        return data_full["date", "returns", "realized_volatility_5", "realized_volatility_11", "realized_volatility_21", "realized_volatility_60"]
    return data_full
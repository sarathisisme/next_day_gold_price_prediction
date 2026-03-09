#!/usr/bin/env python
# coding: utf-8

# In[1]:


from fredapi import Fred
import numpy as np
import yfinance as yf
import pandas as pd
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import warnings
import pickle

warnings.filterwarnings("ignore")

# In[2]:


FRED_API_KEY = "adf5faa6d759eb1f4b335cef60ade574"
START_DATE = "2004-11-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

fred = Fred(api_key=FRED_API_KEY)

# -----------------------------------------
# DOWNLOAD FRED MACRO DATA
# -----------------------------------------
dxy = fred.get_series("DTWEXBGS", observation_start=START_DATE)
real_yield = fred.get_series("DFII10", observation_start=START_DATE)
breakeven = fred.get_series("T10YIE", observation_start=START_DATE)
fed_funds = fred.get_series("DFF", observation_start=START_DATE)
oil = fred.get_series("DCOILWTICO", observation_start=START_DATE)

cpi = fred.get_series("CPIAUCSL", observation_start=START_DATE)
m2 = fred.get_series("M2SL", observation_start=START_DATE)

# -----------------------------------------
# TRANSFORM MONTHLY SERIES
# -----------------------------------------
cpi = cpi.resample("M").last()
m2 = m2.resample("M").last()

cpi_yoy = cpi.pct_change(12) * 100
m2_yoy = m2.pct_change(12) * 100

# Lag monthly macro to avoid look-ahead bias
cpi_yoy = cpi_yoy.shift(1)
m2_yoy = m2_yoy.shift(1)

# -----------------------------------------
# DOWNLOAD MARKET DATA (YAHOO)
# -----------------------------------------
gold = yf.download("GC=F", start=START_DATE, end=END_DATE)["Close"]
sp500 = yf.download("^GSPC", start=START_DATE, end=END_DATE)["Close"]
vix = yf.download("^VIX", start=START_DATE, end=END_DATE)["Close"]

# -----------------------------------------
# COMBINE DAILY DATA
# -----------------------------------------
daily_data = pd.concat(
    [gold, dxy, real_yield, breakeven, fed_funds, oil, sp500, vix],
    axis=1
)

daily_data.columns = [
    "Gold",
    "DXY",
    "Real_Yield_10Y",
    "Breakeven_10Y",
    "Fed_Funds",
    "WTI_Oil",
    "SP500",
    "VIX"
]

START_DATE = "2006-01-01"

# -----------------------------------------
# ADD MONTHLY MACRO (CPI & M2)
# -----------------------------------------
monthly_macro = pd.concat([cpi_yoy, m2_yoy], axis=1)
monthly_macro.columns = ["CPI_YoY", "M2_YoY"]

monthly_macro_daily = monthly_macro.resample("D").ffill()

last_day = gold.index[-1]
monthly_macro_daily = monthly_macro_daily.reindex(
    pd.date_range(start=monthly_macro_daily.index[0], end=last_day, freq="D")
).ffill()

# -----------------------------------------
# FINAL DATASET
# -----------------------------------------
df = pd.concat([daily_data, monthly_macro_daily], axis=1)
df = df.loc[START_DATE:]

# In[3]:


df = df.dropna(subset=['Gold'])

# In[4]:


df = df.ffill()

# In[5]:


df

# In[6]:


df.isna().sum()

# In[7]:


df["log_price"] = np.log(df["Gold"])
df["t"] = np.arange(len(df))

# In[8]:


df_recent = df.tail(252 * 16).copy()

residuals = df_recent["log_price"] - LinearRegression().fit(df_recent[["t"]], df_recent["log_price"]).predict(df_recent[["t"]])
tuned_model = auto_arima(
    residuals,
    stationary=True,
    seasonal=False,
    stepwise=True,
    suppress_warnings=True,
    error_action='ignore',
    max_p=10,
    max_q=10,
    max_d=0,
    trace=True,
    n_jobs=-1
)

# In[9]:


# Save best hyperparameters
p, d, q = tuned_model.order
print("Selected ARIMA order:", (p,d,q))
print("AIC:", tuned_model.aic())

# In[10]:


forecast_horizon = 1
start_index = len(df) - forecast_horizon - 1
WINDOW = 252 * 15

def fit_linear_trend(train_data):
    """Fit a linear trend model to log_price and return model + residuals."""
    X_trend = train_data[["t"]]
    y_trend = train_data["log_price"]
    trend_model = LinearRegression()
    trend_model.fit(X_trend, y_trend)
    train_data = train_data.copy()
    train_data["trend"] = trend_model.predict(X_trend)
    train_data["residual"] = train_data["log_price"] - train_data["trend"]
    return trend_model, train_data

def fit_arma_residual(train_data, p, d, q):
    """Fit an ARIMA model to the residuals and return the fitted model."""
    arma_model = ARIMA(train_data["residual"], order=(p, d, q))
    return arma_model.fit()

def forecast_next_day(trend_model, arma_fitted, last_t):
    """Forecast the next day's gold price using trend + ARMA residual."""
    next_t = np.array([[last_t + 1]])
    trend_forecast = trend_model.predict(next_t)[0]
    residual_forecast = arma_fitted.forecast(steps=1).iloc[0]
    log_price_forecast = trend_forecast + residual_forecast
    return np.exp(log_price_forecast)

def fit_and_save_model(df, start_index, p, d, q, window=WINDOW, save_path="gold_model.pkl"):
    """
    Fit trend + ARMA model on the most recent window ending at start_index.
    Saves both models to a .pkl file and returns the forecast.
    """
    train_data = df.iloc[max(0, start_index - window + 1):start_index + 1]

    trend_model, train_data = fit_linear_trend(train_data)
    arma_fitted = fit_arma_residual(train_data, p, d, q)

    price_forecast = forecast_next_day(trend_model, arma_fitted, train_data["t"].iloc[-1])
    actual = df["Gold"].iloc[start_index + 1]
    date = df.index[start_index + 1]

    build_timestamp = datetime.now().isoformat()

    model_bundle = {
        "trend_model": trend_model,
        "arma_fitted": arma_fitted,
        "last_t": train_data["t"].iloc[-1],
        "build_timestamp": build_timestamp,
    }
    with open(save_path, "wb") as f:
        pickle.dump(model_bundle, f)
    print(f"Model saved to {save_path} (built at {build_timestamp})")

    return date, price_forecast, actual

# --- Usage ---
date, prediction, actual = fit_and_save_model(df, start_index, p, d, q)
print(f"Date: {date} | Forecast: {prediction:.2f} | Actual: {actual:.2f}")
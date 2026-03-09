import streamlit as st
import pickle
import numpy as np
from datetime import datetime

PKL_PATH = "gold_model.pkl"

def forecast_next_day(trend_model, arma_fitted, last_t):
    """Forecast the next day's gold price using trend + ARMA residual."""
    next_t = np.array([[last_t + 1]])
    trend_forecast = trend_model.predict(next_t)[0]
    residual_forecast = arma_fitted.forecast(steps=1).iloc[0]
    log_price_forecast = trend_forecast + residual_forecast
    return np.exp(log_price_forecast)

st.title("Gold Price Forecast")

try:
    with open(PKL_PATH, "rb") as f:
        bundle = pickle.load(f)

    trend_model = bundle["trend_model"]
    arma_fitted = bundle["arma_fitted"]
    last_t = bundle["last_t"]
    build_timestamp = bundle.get("build_timestamp")

    forecast_price = forecast_next_day(trend_model, arma_fitted, last_t)

    st.write(f"**Forecasted Gold Price (next day):** ${forecast_price:,.2f}")

    if build_timestamp:
        formatted = datetime.fromisoformat(build_timestamp).strftime("%B %d, %Y at %I:%M %p")
        st.caption(f"Model last built: {formatted}")
    else:
        st.caption("Model last built: unknown (rebuild to capture timestamp)")

except FileNotFoundError:
    st.error(f"Model file `{PKL_PATH}` not found. Place it in the same folder as this app.")
except Exception as e:
    st.error(f"Error loading model: {e}")
#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import time
import pandas as pd
from datetime import datetime, timedelta
from fredapi import Fred
import numpy as np
import yfinance as yf
import pandas as pd
from xgboost import XGBRegressor
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pytz
import warnings

warnings.filterwarnings("ignore")

# In[2]:


df_sentiment = pd.read_csv("data/sentiment_data_final.csv", index_col=0, parse_dates=True)

# In[3]:


df_sentiment

# In[4]:


API_KEY = "qrZUHabbbGHJWU95326BITrpe1ZX6SbC79MvFmbIKuEICM9l"

MAX_MINUTE_RETRIES = 3   # retry attempts for minute limit
MINUTE_SLEEP = 60        # NYT allows ~5 requests per minute

def fetch_articles(query, begin_date, end_date, page):

    url = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

    params = {
        "api-key": API_KEY,
        "q": query,
        "begin_date": begin_date,
        "end_date": end_date,
        "page": page,
    }

    retries = 0

    while retries <= MAX_MINUTE_RETRIES:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            return data.get("response", {}).get("docs", [])

        if response.status_code == 429:
            retries += 1

            if retries > MAX_MINUTE_RETRIES:
                print("Daily rate limit likely reached.")
                return None  # signal to stop everything

            print(f"Minute rate limit hit. Sleeping {MINUTE_SLEEP}s...")
            time.sleep(MINUTE_SLEEP)
            continue
    
        print("Error:", response.status_code)
        return []

    return None


def get_all_articles(query, start_date, end_date):

    all_articles = []
    current_date = start_date
    stop_collection = False

    while current_date <= end_date and not stop_collection:

        month_end = (current_date.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        if month_end > end_date:
            month_end = end_date

        begin_str = current_date.strftime("%Y%m%d")
        end_str = month_end.strftime("%Y%m%d")

        print(f"Fetching {begin_str} to {end_str}")

        month_articles = []          # store articles for THIS month only
        month_failed = False         # track if any request failed

        for page in range(5):

            docs = fetch_articles(query, begin_str, end_str, page)

            if docs is None:
                month_failed = True  # mark month as incomplete
                stop_collection = True
                break

            if not docs:
                break

            month_articles.extend(docs)
            
            time.sleep(12)

        # Only commit the month if ALL requests succeeded
        if not month_failed:
            all_articles.extend(month_articles)
        else:
            print(f"Skipping incomplete month {begin_str}-{end_str}")

        current_date = month_end + timedelta(days=1)

    print(f"Returning {len(all_articles)} articles collected so far.")
    return pd.json_normalize(all_articles)

# In[5]:


# Run it
start = datetime(2026, 3, 10)

ny_tz = pytz.timezone("America/New_York")
yesterday = datetime.now(ny_tz).replace(hour=0, minute=0, second=0, microsecond=0)
yesterday -= timedelta(days=1)

end = datetime(yesterday.year, yesterday.month, yesterday.day)
print(end)

df = get_all_articles("economy", start, end)

# In[6]:


analyzer = SentimentIntensityAnalyzer()

# In[7]:


# Choose the column containing the text
text_col = 'abstract'

# Drop rows with missing text
df = df.dropna(subset=[text_col, 'pub_date'])

# Compute sentiment
df = df.copy()
df['sentiment'] = df[text_col].apply(
    lambda x: analyzer.polarity_scores(x)['compound']
)

# In[8]:


df['pub_date'] = pd.to_datetime(df['pub_date'])
df['date'] = df['pub_date'].dt.date

# In[9]:


daily_sentiment = df.groupby('date')['sentiment'].mean().reset_index()

# In[10]:


daily_sentiment

# In[11]:


daily_sentiment = daily_sentiment.set_index('date')
daily_sentiment.index = pd.to_datetime(daily_sentiment.index)

# In[12]:


daily_sentiment

# In[13]:


df_sentiment = pd.concat([df_sentiment, daily_sentiment]).sort_index()

# In[14]:


df_sentiment

# In[15]:


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

# In[16]:


START_DATE = "2023-01-01"

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

# In[17]:


df = df.dropna(subset=['Gold'])
df = df.ffill()

# In[18]:


df.isna().sum()

# In[19]:


df = df.join(df_sentiment, how='left')
df.sort_index(inplace=True)
df = df.ffill()

# In[20]:


df.isna().sum()

# In[21]:


df

# In[22]:


TARGET_COL = "Gold"
EXOG_VARS  = [col for col in df.columns if col != TARGET_COL]
GOLD_LAGS  = [1, 2, 3, 4, 5, 6]
EXOG_LAGS  = [1, 2, 3]
TRAIN_WINDOW = 252 * 2  # 504 trading days

XGB_PARAMS = dict(
    n_estimators     = 800,
    max_depth        = 4,
    learning_rate    = 0.01,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1,
    random_state     = 42
)

# ── 1. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df):
    """Build lag features. No target shift needed — we predict from the latest row."""
    df = df.copy()
    df["t"] = np.arange(len(df))

    for lag in GOLD_LAGS:
        df[f"gold_lag_{lag}"] = df[TARGET_COL].shift(lag)

    for col in EXOG_VARS:
        for lag in EXOG_LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df

# ── 2. PREPARE TRAIN SET ──────────────────────────────────────────────────────
def get_train_data(df_feat, train_window):
    """
    Use the last `train_window` rows (excluding the final row) as training data.
    Target = next day's Gold price, so we shift Gold by -1 within this slice.
    """
    train_slice = df_feat.iloc[-(train_window + 1):-1].copy()  # train_window rows
    train_slice["target"] = df_feat[TARGET_COL].iloc[
        -(train_window): len(df_feat)
    ].values  # align next-day gold as target

    train_slice = train_slice.dropna()
    X_train = train_slice.drop(columns=["target"])
    y_train = train_slice["target"]
    return X_train, y_train

# ── 3. PREDICT NEXT DAY ───────────────────────────────────────────────────────
def predict_next_day_gold(df, train_window=TRAIN_WINDOW):
    """
    Train on the last `train_window` rows and predict the next day's Gold price.

    Returns
    -------
    prediction : float
        Predicted Gold price for the next trading day.
    model : XGBRegressor
        The trained model.
    last_date : pd.Timestamp
        The date of the most recent data point used (i.e., today's date in the data).
    """
    if len(df) < train_window + max(GOLD_LAGS):
        raise ValueError(
            f"Need at least {train_window + max(GOLD_LAGS)} rows, got {len(df)}"
        )

    df_feat = build_features(df)

    # ── Training ──────────────────────────────────────────────────────────────
    X_train, y_train = get_train_data(df_feat, train_window)

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # ── Prediction row = the very last row in df_feat ─────────────────────────
    latest_row = df_feat.iloc[[-1]].drop(
        columns=[c for c in ["target"] if c in df_feat.columns],
        errors="ignore"
    )
    latest_row = latest_row[X_train.columns]  # enforce column order

    prediction = model.predict(latest_row)[0]
    last_date  = df_feat.index[-1]

    print(f"Last data date : {last_date.date()}")
    print(f"Next-day Gold prediction : {prediction:.4f}")

    return prediction, model, last_date

# In[23]:


# ── 4. RUN ────────────────────────────────────────────────────────────────────
prediction_tomorrow, model, last_date = predict_next_day_gold(df)

# In[24]:


prediction_tomorrow

# In[25]:


TARGET_COL   = "Gold"
EXOG_VARS    = [col for col in df.columns if col != TARGET_COL]
GOLD_LAGS    = [1, 2, 3, 4, 5, 6]
EXOG_LAGS    = [1, 2, 3]
TRAIN_WINDOW = 252 * 2  # 504 trading days

XGB_PARAMS = dict(
    n_estimators     = 800,
    max_depth        = 4,
    learning_rate    = 0.01,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1,
    random_state     = 42
)

# ── 1. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()
    df["t"] = np.arange(len(df))

    for lag in GOLD_LAGS:
        df[f"gold_lag_{lag}"] = df[TARGET_COL].shift(lag)

    for col in EXOG_VARS:
        for lag in EXOG_LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df

# ── 2. PREPARE TRAIN SET ──────────────────────────────────────────────────────
def get_train_data(df_feat, train_window):
    """
    Train on the last `train_window` rows (excluding the final row).
    Target = same-day Gold price (no shift) — lag features already encode yesterday.
    """
    train_slice = df_feat.iloc[-(train_window + 1):-1].copy()  # train_window rows
    train_slice["target"] = train_slice[TARGET_COL]            # same-day gold as target

    train_slice = train_slice.drop(columns=[TARGET_COL]).dropna()
    X_train = train_slice.drop(columns=["target"])
    y_train = train_slice["target"]
    return X_train, y_train

# ── 3. PREDICT TODAY ──────────────────────────────────────────────────────────
def predict_today_gold(df, train_window=TRAIN_WINDOW):
    """
    Train on the last `train_window` rows and predict today's Gold price.
    The prediction row is the last row of df_feat, with TARGET_COL dropped
    (since today's Gold is what we're predicting).

    Returns
    -------
    prediction : float  — predicted Gold price for today
    model      : XGBRegressor
    today_date : pd.Timestamp — date of the prediction row
    """
    if len(df) < train_window + max(GOLD_LAGS):
        raise ValueError(
            f"Need at least {train_window + max(GOLD_LAGS)} rows, got {len(df)}"
        )

    df_feat = build_features(df)

    # ── Training ──────────────────────────────────────────────────────────────
    X_train, y_train = get_train_data(df_feat, train_window)

    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # ── Prediction row = last row, TARGET_COL dropped (unknown today) ─────────
    latest_row = df_feat.iloc[[-1]].drop(columns=[TARGET_COL], errors="ignore")
    latest_row = latest_row[X_train.columns]  # enforce column order

    prediction = model.predict(latest_row)[0]
    today_date = df_feat.index[-1]

    print(f"Prediction date  : {today_date.date()}")
    print(f"Today's Gold prediction : {prediction:.4f}")

    return prediction, model, today_date

# In[26]:


# ── 4. RUN ────────────────────────────────────────────────────────────────────
prediction_today, model, today_date = predict_today_gold(df)

# ## Next 7 days

# In[27]:


TARGET_COL   = "Gold"
EXOG_VARS    = [col for col in df.columns if col != TARGET_COL]
GOLD_LAGS    = [1, 2, 3, 4, 5, 6]
EXOG_LAGS    = [1, 2, 3]
TRAIN_WINDOW = 252 * 2  # 504 trading days
FORECAST_HORIZON = 7

XGB_PARAMS = dict(
    n_estimators     = 800,
    max_depth        = 4,
    learning_rate    = 0.01,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    reg_alpha        = 0.1,
    reg_lambda       = 1,
    random_state     = 42
)

# ── 1. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df):
    df = df.copy()
    df["t"] = np.arange(len(df))

    for lag in GOLD_LAGS:
        df[f"gold_lag_{lag}"] = df[TARGET_COL].shift(lag)

    for col in EXOG_VARS:
        for lag in EXOG_LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df

# ── 2. PREPARE TRAIN SET ──────────────────────────────────────────────────────
def get_train_data(df_feat, train_window):
    """Target = next day's Gold price (shift -1)."""
    train_slice = df_feat.iloc[-(train_window + 1):-1].copy()
    train_slice["target"] = df_feat[TARGET_COL].iloc[
        -(train_window): len(df_feat)
    ].values

    train_slice = train_slice.dropna()
    # Drop raw price columns — only lag features + "t" should remain
    cols_to_drop = ["target", TARGET_COL] + EXOG_VARS
    X_train = train_slice.drop(columns=[c for c in cols_to_drop if c in train_slice.columns])
    y_train = train_slice["target"]
    return X_train, y_train

# ── 3. RECURSIVE 7-DAY FORECAST ───────────────────────────────────────────────
def predict_next_7_days(df, train_window=TRAIN_WINDOW, horizon=FORECAST_HORIZON):
    """
    Recursive multi-step forecast for the next `horizon` trading days.

    Strategy:
      - Train once on the last `train_window` rows (next-day target).
      - At each step, build a feature row from the rolling gold price buffer
        and the last known exog lags, then feed the prediction back into the
        buffer for the next step.

    Returns
    -------
    forecast_df : pd.DataFrame
        DataFrame with columns ['date', 'predicted_gold'] for each forecast day.
    model : XGBRegressor
    """
    if len(df) < train_window + max(GOLD_LAGS):
        raise ValueError(
            f"Need at least {train_window + max(GOLD_LAGS)} rows, got {len(df)}"
        )

    df_feat = build_features(df)

    # ── Train once ────────────────────────────────────────────────────────────
    X_train, y_train = get_train_data(df_feat, train_window)
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)

    # ── Rolling gold buffer: recent known prices for lag construction ─────────
    # Keep max(GOLD_LAGS) most recent gold prices; new predictions append here
    gold_buffer = list(df[TARGET_COL].iloc[-max(GOLD_LAGS):].values)

    # Last known exog lag values (taken from the last feature row)
    last_feat_row = df_feat.iloc[-1]

    last_t     = int(last_feat_row["t"])
    predictions = []
    forecast_dates = pd.bdate_range(                          # business days only
        start=df.index[-1] + pd.offsets.BDay(1), periods=horizon
    )

    for step in range(horizon):
        # Build feature row for this step
        feat = {}
        feat["t"] = last_t + step + 1

        # Gold lags — pulled from the rolling buffer
        for lag in GOLD_LAGS:
            feat[f"gold_lag_{lag}"] = gold_buffer[-(lag)]

        # Exog lags — held constant at last known values (forward-fill assumption)
        # For lag_1 → use lag_2's value from previous step, etc. (shift forward)
        for col in EXOG_VARS:
            for lag in EXOG_LAGS:
                feat[f"{col}_lag_{lag}"] = last_feat_row.get(f"{col}_lag_{lag}", np.nan)

        # Align to training columns and predict
        feat_df = pd.DataFrame([feat])[X_train.columns]
        pred    = model.predict(feat_df)[0]

        predictions.append(pred)
        gold_buffer.append(pred)   # feed prediction back as next lag

    # ── Results ───────────────────────────────────────────────────────────────
    forecast_df = pd.DataFrame({
        "date"           : forecast_dates,
        "predicted_gold" : predictions
    }).set_index("date")

    print(f"Last data date : {df.index[-1].date()}")
    print(f"\n{'Date':<15}  {'Predicted Gold':>15}")
    print("-" * 32)
    for date, row in forecast_df.iterrows():
        print(f"{str(date.date()):<15}  {row['predicted_gold']:>15.4f}")

    return forecast_df, model


# ── 4. RUN ────────────────────────────────────────────────────────────────────
forecast_df, model = predict_next_7_days(df)

# In[28]:


# ── 4. RUN ────────────────────────────────────────────────────────────────────
forecast_df, model = predict_next_7_days(df)

# In[31]:


forecast_df

# In[43]:


new_row = pd.DataFrame({
    'predicted_gold': [prediction_today]
}, index=[pd.Timestamp(today_date)])

forecast_df_final = pd.concat([new_row, forecast_df])

# In[44]:


forecast_df_final

# In[48]:


forecast_df_final.iloc[1, forecast_df_final.columns.get_loc('predicted_gold')] = prediction_tomorrow

# In[49]:


forecast_df_final

# In[50]:


forecast_df_final.to_csv('gold_predictions.csv', index=True)

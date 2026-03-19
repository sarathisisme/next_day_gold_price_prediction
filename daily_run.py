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


TARGET_COL   = "Gold"
EXOG_VARS  = [col for col in df.columns if col != TARGET_COL]
GOLD_LAGS    = [1, 2, 3, 4, 5, 6]
EXOG_LAGS    = [1, 2, 3]
TRAIN_WINDOW = 252 * 2   # 3780 trading days
TEST_WINDOW  = 10

XGB_PARAMS = dict(
    n_estimators    = 800,
    max_depth       = 4,
    learning_rate   = 0.01,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 1,
    random_state    = 42
)

# In[23]:


# ── 1. FEATURE ENGINEERING ────────────────────────────────────────────────────
def build_features(df):
    """Build all features exactly as specified."""
    df = df.copy()

    # Next day target
    df["target"] = df[TARGET_COL].shift(-1)
    
    # Integer time index as feature
    df["t"] = np.arange(len(df))

    # Gold lags
    for lag in GOLD_LAGS:
        df[f"gold_lag_{lag}"] = df[TARGET_COL].shift(lag)

    # Exogenous lags
    for col in EXOG_VARS:
        for lag in EXOG_LAGS:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)

    return df

# ── 2. ROLLING FORECAST (REFACTORED) ─────────────────────────────────────────

def validate_data(df_feat, train_window, test_window):
    """Validate enough data exists for the rolling forecast."""
    total_required = train_window + test_window
    if len(df_feat) < total_required:
        raise ValueError(f"Need {total_required} rows after feature engineering, got {len(df_feat)}")
    return df_feat.iloc[-(train_window + test_window):]


def get_window_split(df_feat, i, train_window):
    """Slice and split a single rolling window into train and latest row."""
    window     = df_feat.iloc[i : i + train_window].copy()
    latest_row = window.iloc[-1:].copy()
    train_df   = window.iloc[:-1].dropna()
    return train_df, latest_row


def get_X_y(train_df):
    """Split training window into features and target."""
    X_train = train_df.drop("target", axis=1)
    y_train = train_df["target"]
    return X_train, y_train


def train_model(X_train, y_train):
    """Train an XGBRegressor on the given data."""
    model = XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train, verbose=False)
    return model


def predict_next_day(model, latest_row, X_train):
    """Generate next day prediction from the latest row."""
    latest_features = latest_row.drop("target", axis=1)
    latest_features = latest_features[X_train.columns]    # ensure column order
    return model.predict(latest_features)[0]


def run_single_step(df_feat, i, train_window):
    """Run a single rolling step: split → train → predict."""
    train_df, latest_row = get_window_split(df_feat, i, train_window)
    X_train, y_train     = get_X_y(train_df)
    model                = train_model(X_train, y_train)
    pred                 = predict_next_day(model, latest_row, X_train)
    return pred, model


def rolling_xgb_forecast(df, train_window=TRAIN_WINDOW, test_window=TEST_WINDOW):
    """
    Rolling 1-step ahead XGBoost forecast.

    At each step i:
      - Train on rows [i : i + train_window - 1]  (train_window rows)
      - 'latest_row' is row [i + train_window - 1] (last row of window)
      - Predict target = row [i + train_window]    (true next day)
    """
    df_feat = build_features(df)
    df_feat = validate_data(df_feat, train_window, test_window)

    predictions, actuals, pred_dates = [], [], []

    print(f"Train window : {train_window} days")
    print(f"Test window  : {test_window} days")
    print(f"Rolling forecast started...\n")

    for i in range(test_window):

        pred, model = run_single_step(df_feat, i, train_window)

        predictions.append(pred)
        actuals.append(df_feat.iloc[i + train_window][TARGET_COL])
        pred_dates.append(df_feat.index[i + train_window])

        print(f"\r  Progress: {((i + 1) / test_window) * 100:.1f}%", end="", flush=True)

    return pred_dates, predictions, actuals, model

# In[24]:


# ── 6. RUN & EVALUATE ─────────────────────────────────────────────────────────
pred_dates, predictions, actuals, final_model = rolling_xgb_forecast(df)

# In[27]:


predictions

# In[26]:


def predict_true_tomorrow(df, train_window=TRAIN_WINDOW):
    """
    Train on the last train_window rows of the full df and predict
    the one day that has not yet occurred.
    
    latest_row = df.iloc[-1] (today — last known data point)
    prediction = tomorrow's Gold price (genuinely unknown)
    """
    df_feat = build_features(df)  # built on full df, no validate_data trimming

    # Window = last train_window rows of full df
    window     = df_feat.iloc[-train_window:].copy()
    latest_row = window.iloc[-1:].copy()          # today = iloc[-1]
    train_df   = window.iloc[:-1].dropna()        # train on iloc[-train_window:-1]

    X_train, y_train = get_X_y(train_df)
    model            = train_model(X_train, y_train)
    pred             = predict_next_day(model, latest_row, X_train)

    tomorrow_date = df_feat.index[-1] + pd.offsets.BDay(1)

    print(f"Last known date            : {df_feat.index[-1].date()}")
    print(f"Prediction date (tomorrow) : {tomorrow_date.date()}")
    print(f"Tomorrow's Gold prediction : {pred:.4f}")
    return pred, tomorrow_date


# ── RUN ───────────────────────────────────────────────────────────────────────
pred_dates, predictions, actuals, model = rolling_xgb_forecast(df)
tomorrow_pred, tomorrow_date = predict_true_tomorrow(df)

# In[29]:


# ── BUILD RESULTS DATAFRAME ───────────────────────────────────────────────────
results_df = pd.DataFrame({
    "prediction" : predictions + [tomorrow_pred],
    "actual"     : actuals     + [np.nan],
}, index=pd.DatetimeIndex(pred_dates + [tomorrow_date]))

results_df.index.name = "date"

results_df.to_csv("gold_predictions.csv")

# In[30]:


results_df

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("🥇 Gold Price Predictions")

st.image("https://upload.wikimedia.org/wikipedia/commons/2/22/1000g-Goldbarren-Umicore.jpg",
         use_container_width=True)

st.subheader("Next Day Forecast (Closing price of the day)")

df = pd.read_csv("gold_predictions.csv", index_col=0, parse_dates=True)

y_min = df[['actual', 'prediction']].min().min() * 0.999
y_max = df[['actual', 'prediction']].max().max() * 1.001

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['prediction'],
    mode='lines+markers',
    name='Prediction',
    line=dict(color='gold', width=2),
    marker=dict(size=8)
))
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['actual'],
    mode='lines+markers',
    name='Actual',
    line=dict(color='steelblue', width=2),
    marker=dict(size=8)
))
fig.update_layout(
    title="Gold Price Forecast",
    xaxis_title="Date",
    yaxis_title="Predicted Price (USD per oz)",
    yaxis=dict(range=[y_min, y_max]),
    xaxis=dict(tickmode="array", tickvals=df.index[::2], tickformat="%Y-%m-%d"),
    hovermode="x unified"
)
fig.add_trace(go.Scatter(
    x=df.index[-1:],
    y=df['prediction'].iloc[-1:],
    mode='markers',
    name='Tomorrow',
    marker=dict(color='red', size=14, symbol='star')
))
st.plotly_chart(fig, use_container_width=True)
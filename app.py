import streamlit as st
import pandas as pd
import plotly.graph_objects as go

page = st.sidebar.radio("Navigate", ["🥇 Gold Prices Forecast", "🥇 Gold Prices History"])

if page == "🥇 Gold Prices Forecast":
    st.title("🥇 Gold Price Predictions")

    st.image("https://upload.wikimedia.org/wikipedia/commons/2/22/1000g-Goldbarren-Umicore.jpg",
             use_container_width=True)

    st.subheader("Next 3 Days Forecast (Closing price of the day)")

    df = pd.read_csv("gold_predictions.csv", index_col=0, parse_dates=True)
    df = df[1:-4].head(7)

    price_col = df.columns[0]
    y_min = df[price_col].min() * 0.999
    y_max = df[price_col].max() * 1.001

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[price_col],
        mode='lines+markers',
        line=dict(color='gold', width=2),
        marker=dict(size=8)
    ))
    fig.update_layout(
        title="Gold Price Forecast",
        xaxis_title="Date",
        yaxis_title="Predicted Price (USD per oz)",
        yaxis=dict(range=[y_min, y_max]),
        xaxis=dict(tickmode="array", tickvals=df.index, tickformat="%Y-%m-%d"),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

elif page == "🥇 Gold Prices History":
    st.title("🥇 Gold Prices 10 day History")

    st.image("https://upload.wikimedia.org/wikipedia/commons/3/34/400-oz-Gold-Bars-AB-01.jpg",
             use_container_width=True)

    st.subheader("Gold Prices 10 day History (Closing price of the day)")

    df = pd.read_csv("previous_predictions.csv", index_col=0, parse_dates=True)

    y_min = df[['predictions', 'actuals']].min().min() * 0.999
    y_max = df[['predictions', 'actuals']].max().max() * 1.001

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df.index, y=df['predictions'],
        mode='lines+markers', name='Prediction',
        line=dict(color='silver', width=2), marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df['actuals'],
        mode='lines+markers', name='Actual',
        line=dict(color='steelblue', width=2), marker=dict(size=8)
    ))


    fig.update_layout(
        title="Gold Price Prediction History",
        xaxis_title="Date",
        yaxis_title="Gold Price (USD per oz)",
        yaxis=dict(range=[y_min, y_max]),
        xaxis=dict(
            tickmode="array",
            tickvals=df.index[::2],  # show every other date
            tickformat="%Y-%m-%d",
            tickangle=45
        ),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os

# Function to fetch stock data
def fetch_stock_data(symbol, start_date, end_date):
    try:
        stock_data = yf.download(symbol, start=start_date, end=end_date)
        if stock_data.empty:
            return None
        return stock_data
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return None

# Function to analyze stock data against event data
def analyze_stock_data(stock_data, dates, event_data):
    analysis = []
    previous_close = None
    previous_event_rate = None
    for date in dates:
        if date in stock_data.index:
            close_price = stock_data.loc[date]['Close']
            event_rate = event_data[date]
            stock_percentage_change = ((close_price - previous_close) / previous_close) * 100 if previous_close else None
            event_percentage_change = ((event_rate - previous_event_rate) / previous_event_rate) * 100 if previous_event_rate else None
            analysis.append({
                'Date': date, 'Close Price': close_price, 'Event Rate': event_rate,
                'Stock Percentage Change': stock_percentage_change, 'Event Percentage Change': event_percentage_change
            })
            previous_close = close_price
            previous_event_rate = event_rate
    return analysis

# Function to plot stock analysis
def plot_stock_analysis(symbol, analysis, company_name):
    dates = [entry['Date'] for entry in analysis]
    stock_changes = [entry['Stock Percentage Change'] for entry in analysis]
    event_changes = [entry['Event Percentage Change'] for entry in analysis]

    plt.figure(figsize=(10, 6))
    plt.plot(dates, stock_changes, marker='o', label='Stock Monthly % Change')
    plt.plot(dates, event_changes, marker='x', label='Event Monthly % Change')
    plt.xlabel('Date')
    plt.ylabel('Percentage Change')
    plt.title(f'{company_name} ({symbol}) - Monthly Changes')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return plt

# Function to predict future stock trends
def predict_stock_trends(data):
    data['Date'] = pd.to_datetime(data['Date'])
    data['YearMonth'] = data['Date'].dt.to_period('M')
    data['Score'] = data['Change (%)']
    
    score_agg = data.groupby(['Stock Symbol', 'Company Name', 'YearMonth'])['Score'].sum().reset_index()
    total_scores = score_agg.groupby(['Stock Symbol', 'Company Name'])['Score'].sum().reset_index()

    latest_month = score_agg['YearMonth'].max()
    future_dates = pd.date_range(start=latest_month.to_timestamp() + pd.DateOffset(months=1), periods=3, freq='M')
    future_periods = future_dates.to_period('M')

    future_results = []
    for future_period in future_periods:
        future_scores = total_scores.copy()
        future_scores['Score'] = np.random.uniform(-10, 10, len(future_scores))  # Example random scores
        future_scores['YearMonth'] = future_period
        future_results.append(future_scores)

    return pd.concat(future_results)

# Streamlit UI
st.title("Stock Analysis and Prediction App")

# Upload files
symbols_file = st.file_uploader("Upload Symbols.xlsx", type=["xlsx"])
event_file = st.file_uploader("Upload event.xlsx", type=["xlsx"])
stock_changes_file = st.file_uploader("Upload Stock_Changes_By_Date.xlsx", type=["xlsx"])

if symbols_file and event_file:
    symbols_df = pd.read_excel(symbols_file)
    event_df = pd.read_excel(event_file)
    
    if 'StockSymbol' in symbols_df.columns and 'Date' in event_df.columns and 'Rate' in event_df.columns:
        event_df['Date'] = pd.to_datetime(event_df['Date']).dt.strftime('%Y-%m-%d')
        event_data = dict(zip(event_df['Date'], event_df['Rate']))
        dates = list(event_data.keys())

        st.subheader("Fetching Stock Data")
        stock_data_dict = {}
        for index, row in symbols_df.iterrows():
            symbol = row['StockSymbol']
            stock_data = fetch_stock_data(symbol, dates[0], dates[-1])
            if stock_data is not None:
                stock_data_dict[symbol] = stock_data
        
        st.subheader("Analyzing Stock Data")
        for symbol, stock_data in stock_data_dict.items():
            analysis = analyze_stock_data(stock_data, dates, event_data)
            plt = plot_stock_analysis(symbol, analysis, symbol)
            st.pyplot(plt)
    
    else:
        st.error("Invalid file structure. Ensure correct column names.")

if stock_changes_file:
    stock_changes_df = pd.read_excel(stock_changes_file)
    if {'Date', 'Stock Symbol', 'Company Name', 'Change (%)'}.issubset(stock_changes_df.columns):
        st.subheader("Predicting Future Stock Trends")
        future_trends = predict_stock_trends(stock_changes_df)
        st.dataframe(future_trends)
    else:
        st.error("Invalid file structure. Ensure correct column names.")

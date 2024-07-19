import yfinance as yf
import pandas as pd

def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

def preprocess_data(data):
    data.dropna(inplace=True)
    return data

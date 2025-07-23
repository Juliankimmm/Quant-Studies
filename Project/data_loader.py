import yfinance as yf
import pandas as pd

def get_data(ticker, period="2y", interval="1d"):
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
    df.dropna(inplace=True)
    return df
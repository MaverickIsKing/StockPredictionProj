import pandas as pd
import numpy as np

# For reading stock data from yahoo
# import yfinance as yf

# For time stamps
from datetime import datetime


def calculate_ma(company, p):
    column_name = f"MA for {p} days"
    company_r = company.copy()
    company_r[column_name] = company['Adj Close'].rolling(window=p).mean()
    return company_r

def pct_change(company, p):
    column_name = f"Pct_change for {p} days"
    company_r = company.copy()
    company_r[column_name] = company['Adj Close'].pct_change(periods=p, fill_method=None)
    return company_r

def log_change(company, p):
    column_name = f"Log_change for {p} days"
    company_r = company.copy()
    company_r[column_name] = np.log(company['Adj Close'] / company['Adj Close'].shift(p))
    return company_r


def rolling_vol(company, p):
    column_name = f"Rolling_vol for {p} days"
    company_r = company.copy()
    company_r['Daily return'] = company['Adj Close'].pct_change(fill_method=None)
    company_r[column_name] = company_r['Daily return'].rolling(window=p).std()
    company_r["Annualized_vol"] = company_r[column_name] * np.sqrt(252)
    return company_r

def calculate_ema(company, p):
    column_name = f"EMA for {p} days"
    company_r = company.copy()
    company_r[column_name] = company['Adj Close'].ewm(span = p).mean()
    return company_r

def calculate_macd(company):
    return 0


def calculate_rsi(company):
    return 0

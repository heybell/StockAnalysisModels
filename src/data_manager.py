import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler

# 분기별 주가 데이터 가져오기 : yfinance
def load_stock_data(ticker, start_date, end_date):
    daily_data = yf.download(ticker, start=start_date, end=end_date)
    # quarterly_data = daily_data.resample('Q').last()

    return daily_data

# 분기별 지수 데이터 가져오기 : csv 파일
def load_index_data(file_path):
    index_data = pd.read_csv(file_path)
    index_data['Date'] = pd.to_datetime(index_data['Date'])

    return index_data

# 주가와 지수 데이터 합치기 : 날짜 기준
def merge_data(stock_data, index_data):
    merged_data = pd.merge(stock_data, index_data, on='Date')

    return merged_data

# 데이터 정규화 : MinMaxScaler
def scale_min_max(data):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    return scaled_data
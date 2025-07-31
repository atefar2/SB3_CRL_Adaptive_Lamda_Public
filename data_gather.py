import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from collections import Counter
from math import ceil, floor 
import pandas as pd
import pandas_ta as pdta
import config
import time
import schedule
from datetime import datetime
from datetime import date
from datetime import timedelta
import numpy as np
import statsmodels.api as sm
import scipy as sp
from scipy.stats import theilslopes , linregress
from scipy.ndimage import gaussian_filter1d , generic_filter1d
import scipy.signal as signal
import scipy.interpolate as si
from scipy.signal import find_peaks
from scipy import stats
import pywt #https://pywavelets.readthedocs.io/en/latest/ref/index.html
import mplcursors
import pickle
# import gtda
# from gtda.time_series import SingleTakensEmbedding
# from gtda.homology import VietorisRipsPersistence
# from gtda.plotting import plot_diagram
# from groq import Groq
# import anthropic
# import pprint
# from sklearn.linear_model import TheilSenRegressor

# import networkx as nx

# import matplotlib.dates as mdates
# import alpaca
import ta as ta
import pytz
import datetime
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.trading.requests import MarketOrderRequest ,LimitOrderRequest, ClosePositionRequest
from alpaca.data.requests import StockLatestQuoteRequest, CryptoLatestQuoteRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
from IPython.display import display, clear_output
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas_ta as pdta
import config
import os
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide,OrderClass,OrderType, QueryOrderStatus, OrderStatus

from sklearn.preprocessing import MinMaxScaler
count =0
alt_pkl_file_path =0
if alt_pkl_file_path==0:
    trading_client = TradingClient(config.alpaca_paper_api_key, config.alpaca_paper_secret_key, paper=True)
    stock_client = StockHistoricalDataClient(config.alpaca_paper_api_key,  config.alpaca_paper_secret_key)
    api = REST(config.alpaca_paper_api_key, config.alpaca_paper_secret_key,base_url=config.alpaca_paper_endpoint)
elif alt_pkl_file_path==1:
    trading_client = TradingClient(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key, paper=True)
    stock_client = StockHistoricalDataClient(config.alpaca_paper_counter_api_key,  config.alpaca_paper_counter_secret_key)
    api = REST(config.alpaca_paper_counter_api_key, config.alpaca_paper_counter_secret_key,base_url=config.alpaca_paper_counter_endpoint)
elif alt_pkl_file_path==2:
    trading_client = TradingClient(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key, paper=True)
    stock_client = StockHistoricalDataClient(config.alpaca_paper_mega_api_key,  config.alpaca_paper_mega_secret_key)
    api = REST(config.alpaca_paper_mega_api_key, config.alpaca_paper_mega_secret_key,base_url=config.alpaca_paper_mega_endpoint)

account = trading_client.get_account()
account
print(account)
# crypto_client = CryptoHistoricalDataClient(config.alpaca_paper_api_key,  config.alpaca_paper_secret_key)

###TIME SENSITIVITY###
time_sensev = 0.50
time_incriment = 1 # 10
time_incriment5 = 10
# if count == 0:
#     time_incriment =1
#     print('count==0',count)

# # else:
# #     time_incriment =2
# #     print('slower',time_incriment)
# elif count > 0 :
#     if (abs(bbma_trading_final.iloc[-1]> 0).sum() / len(bbma_trading_final.iloc[-1])) > time_sensev and (abs(vol_anomaly.iloc[-1]>1.75).sum() / len(vol_anomaly.iloc[-1])) > time_sensev:
#         print(count)
#         time_incriment =1
#         print('faster',time_incriment)
#     else:
#         time_incriment =2
#         print('else slower',time_incriment)

# symbol = tickers_L #combined_tickers #combined_tickers_unique #tickers_L 
symbol = ['TQQQ','SQQQ']# tickers_L #['TQQQ,SQQQ,NVDA,USO,TSLA,SCO,UNH,YINN,YANG,NUGT,DUST,UVXY,KOLD,BOIL,LABU,LABD,TNA,TZA,FAS,FAZ,SOXL,SOXS,ERY,ERX,']

print('time_incriment: ',time_incriment)
print('count;',count)
if count >0:
    print('faster count: ',abs(bbma_trading_final.iloc[-1]> 0).sum())
    print('faster count: ',(abs(bbma_trading_final.iloc[-1]> 0).sum() / len(bbma_trading_final.iloc[-1])) > time_sensev and (abs(vol_anomaly.iloc[-1]>1.75).sum() / len(vol_anomaly.iloc[-1])) > time_sensev)


timeframeunit = TimeFrameUnit.Minute #TimeFrameUnit.Day

# Get the market calendar
calendar = api.get_calendar()
now = datetime.datetime.utcnow()
# Get today's date
today = datetime.datetime.today().date()

# Find the last trading day
last_trading_day = None

for day in reversed(calendar):
    if pd.Timestamp(day.date).date() < today:  # convert Timestamp to date
        last_trading_day = pd.Timestamp(day.date, hour=12, tz='UTC')
        break

# Current time
now = datetime.datetime.now(pytz.UTC)

# Calculate the difference in hours
hours_since_last_noon = int((now - last_trading_day).total_seconds() / 3600)-16

print(last_trading_day)
print(hours_since_last_noon)

if str(datetime.datetime.utcnow().time()) < '13:29:59.000000':
    lookbak_time = 24
elif '08:00:00.971400' < str(datetime.datetime.utcnow().time()) <'16:00:00.971400'  and str(datetime.datetime.utcnow().time()) <'20:00:00.971400':
    lookbak_time = 12
elif '16:00:00.971400' < str(datetime.datetime.utcnow().time())  and str(datetime.datetime.utcnow().time()) <'20:00:00.971400':
    lookbak_time = 16
elif str(datetime.datetime.utcnow().time()) > '20:00:00.971400':
    lookbak_time = 20
    # print('lookbak_time',lookbak_time)
else:
    # lookbak_time = 24
    pass
print('lookbak_time',lookbak_time)
look_back_days = 0
look_back_days_vertiacl = 0

# multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=symbol)
# lookback_days = 100 + 
all_df_open = []
for i in range(1, 3, 1):
    print(i)
    look_back_days = i * 100
    print(f"look_back_days: {look_back_days}")
    print(f"\n")
    Histoic_start = datetime.datetime.today()-timedelta(days=look_back_days) 
    print(f"Histoic_start: {Histoic_start}")
    if i > 0:
        end_time = datetime.datetime.today() - timedelta(days=look_back_days-100)
    else:
        end_time = datetime.datetime.today() - timedelta(days=0)
    print(f"end_time: {end_time}")
    print(f"\n")

    # Histoic_start = datetime.datetime.today()-timedelta(days=100) 
    # start = datetime.datetime.now(pytz.utc)-timedelta( days=look_back_days,hours= float(lookbak_time))#hours_since_last_noon)+16)#24 #days=365)# #(2023,1,1)#hours_since_last_noon
    # starter = datetime.datetime.now(pytz.utc)-timedelta(days=look_back_days,hours= float(lookbak_time))#hours_since_last_noon)+16)#24
    # start2 = datetime.datetime.now(pytz.utc)-timedelta(days=3, hours= float(lookbak_time))#hours_since_last_noon)+16)#24 #days=365)# #(2023,1,1)#hours_since_last_noon
    # reststart2 = start2.strftime("%Y-%m-%dT%H:%M:%SZ")    
    # # Then apply the timedelta to the current time with 1 day back
    # end_time = datetime.datetime.now(pytz.utc) - timedelta(days=look_back_days)
    # # Now apply replace to the datetime object, not the timedelta
    # end = end_time #.replace(hour=20, minute=0, second=0, microsecond=0)

    request_params = StockBarsRequest(symbol_or_symbols=symbol,start=Histoic_start, end=end_time, timeframe=TimeFrame(time_incriment,timeframeunit)) # use singal symblos til solving how to work with multi symbol manipulation
    historic = stock_client.get_stock_bars(request_params)
    historic = historic.df
    historic = historic.bfill()#.bfill()#.bfill()#.iloc[-look_back:] #.iloc[:(len(bars)-2)]
    historic.reset_index(inplace=True)
    historic['timestamp'] = pd.to_datetime(historic['timestamp'], unit='ms',utc=True)
    historic = historic.set_index('timestamp')

    historic =historic.drop_duplicates(keep='last')
    historic = historic.pivot(columns='symbol',values=historic.columns)

    fill_mapping = {column_name: column_name for column_name in historic.symbol.columns}
    # historic['symbol'] = historic.symbol.apply(lambda col: col.fillna(fill_mapping[col.name]))
    # historic = historic.drop(columns=['symbol'])

    historic.tail(50)


    # Filter for market open hours (9:30 to 16:00) datetime.datetime.strptime("13:30", "%H:%M").time())
    is_open = (historic.index.time >= datetime.datetime.strptime("13:30", "%H:%M").time()) & \
            (historic.index.time <= datetime.datetime.strptime("20:00", "%H:%M").time())
    df_open = historic[is_open]

    all_df_open.append(df_open)

df_multiday = pd.concat(all_df_open)
df_multiday.sort_index(inplace=True)

# Create a continuous x-axis: "market minute index"
df_multiday = df_multiday.copy()
df_multiday['day'] = df_multiday.index.date

# Count minutes since the start of the first day (stitching all open periods together)
minute_offsets = []
offset = 0
for day, group in df_multiday.groupby('day'):
    n = len(group)
    minute_offsets.extend(range(offset, offset + n))
    offset += n

df_multiday['market_minute'] = minute_offsets
df_multiday.to_pickle("df_multiday.pkl")
print(df_multiday)

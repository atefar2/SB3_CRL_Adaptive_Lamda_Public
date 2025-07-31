# %% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import requests
import json
import time
import random
import re
import datetime
import pytz
import pickle


preprocessed_data = pd.read_csv('cleaned_preprocessed.csv')

# print(preprocessed_data.head())
# print(preprocessed_data.info())
# print(preprocessed_data.describe())
# print(preprocessed_data.columns)
# print(preprocessed_data.shape)
# print(preprocessed_data.dtypes)
# print(preprocessed_data.isnull().sum())
print(preprocessed_data)

print(f"-----------------")
# print(preprocessed_data.date.tail(50))


# %%
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import requests
import json
import time
import random
import re
import datetime
import pytz
import pickle
import gs_quant 
import yfinance as yf
import pandas_datareader as pdr
import datetime
from datetime import datetime, timedelta


start_date = datetime.utcnow() - timedelta(days=3)
end_date = datetime.utcnow()
symbol = "AAPL"

df = yf.download(symbol, start=start_date, end=end_date, interval='1m',)
print(df)
plt.figure(figsize=(10, 5))
plt.plot(df.index, df.Close)
plt.show()


# %%
!pip install yfinance
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, timedelta

# Download 1-minute data for the last 2 days
# start_date = datetime.utcnow() - timedelta(days=2)
# end_date = datetime.utcnow()
length = 8
start_date = datetime.utcnow() - timedelta(days=length)
end_date = datetime.utcnow()
symbol = "MMM"

df = yf.download(symbol, start=start_date, end=end_date, interval='1m', progress=False)
df = df.tz_convert('US/Eastern')

# Filter for market open hours (9:30 to 16:00)
is_open = (df.index.time >= datetime.strptime("0:00", "%H:%M").time()) & \
          (df.index.time <= datetime.strptime("20:00", "%H:%M").time())
df_open = df[is_open]

# Create a continuous x-axis: "market minute index"
df_open = df_open.copy()
df_open['day'] = df_open.index.date

# Count minutes since the start of the first day (stitching all open periods together)
minute_offsets = []
offset = 0
for day, group in df_open.groupby('day'):
    n = len(group)
    minute_offsets.extend(range(offset, offset + n))
    offset += n

df_open['market_minute'] = minute_offsets

# Plot
plt.figure(figsize=(14, 6))
plt.plot(df_open['market_minute'], df_open['Close'], color='b')

# Optionally, add vertical lines to indicate day changes
for day, group in df_open.groupby('day'):
    plt.axvline(group['market_minute'].iloc[0], color='gray', linestyle='--', alpha=0.5)
    plt.text(group['market_minute'].iloc[0], plt.ylim()[1], str(day), va='bottom', ha='left', fontsize=8, color='gray')

plt.xlabel("Market Minutes (stitched, no overnight gaps)")
plt.ylabel("AAPL Close Price")
plt.title("AAPL Intraday Prices (Market Hours Only, No Gaps Between Days)")
plt.tight_layout()
plt.show()

print(len(df_open))

# %%

631587/2302 

# %%

import os 
import pandas as pd
text_file = open("all_etf.txt", "r")
etf_ticker = text_file.read()
text_file.close()
etf_ticker = etf_ticker.split("\n")
etf_ticker = [ticker.strip() for ticker in etf_ticker if ticker.strip()]

# Extract only the ticker symbols (first column before tab)
ticker_symbols = []
for line in etf_ticker:
    if line.strip():  # Skip empty lines
        symbol = line.split("\t")[0].strip()  # Split by tab and take first part
        ticker_symbols.append(symbol)
        print(symbol)

print(f"\nTotal ETF symbols extracted: {len(ticker_symbols)}")







# %%

# login to Trade Account with login()
# make sure that trade server is enabled in MT5 client terminal

login = 97070354
password = 'iIeElL0176_md.'
server = 'XMGlobal-MT5 5'

symbols = ['EURUSD',
           'GBPUSD',
           'USDCAD',
           'USDCHF',
           'USDJPY',
           'AUDUSD',
           ]

import os
import MetaTrader5 as mt  # pip install MetaTrader5
import pandas as pd  # pip install pandas
from datetime import datetime, timedelta

# from .constants import symbols, timeframes, start_time
# from .constants import server, password, login

# start the platform with initialize()
mt.initialize()

if mt.login(login, password, server) == True:
    print('Connected\n')
else:
    print('MT5 connection Failed\n')

print('Starting to download login details\n')
# get account info
account_info = mt.account_info()
print(account_info)

# getting specific account data
login_number = account_info.login
balance = account_info.balance
equity = account_info.equity

print('\nlogin: ', login_number, 'balance: ', balance, 'equity: ', equity)

print('Data downloading started')


# os.makedirs(os.path.dirname('Data'),exist_ok=True)

for i in symbols:
        ohlc_data_h1 = pd.DataFrame(mt.copy_rates_range(i, mt.TIMEFRAME_H1, datetime(2015, 1, 1), datetime.now()))
        print('Download complete for ', i, ' for H1')
        ohlc_data_h4 = pd.DataFrame(mt.copy_rates_range(i, mt.TIMEFRAME_H4, datetime(2015, 1, 1), datetime.now()))
        print('Download complete for ', i, ' for H4')
        file_name_h1 = 'H1' + '_' + i + '_' + '2015' + '.csv'
        file_name_h4 = 'H1' + '_' + i + '_' + '2015' + '.csv'
        ohlc_data_h1.to_csv(file_name_h1,index=False,header=True)
        ohlc_data_h4.to_csv(file_name_h4,index=False,header=True)

print('Download complete')
mt.close()
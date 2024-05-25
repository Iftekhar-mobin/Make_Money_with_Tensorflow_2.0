import MetaTrader5 as mt  # pip install MetaTrader5
import pandas as pd  # pip install pandas
from datetime import datetime, timedelta

from .constants import symbols, timeframes, start_time
from .constants import server, password, login

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

for i in range(len(symbols)):
    for j in range(len(timeframes)):
        ohlc_data = pd.DataFrame(mt.copy_rates_range(i, mt.j, datetime(start_time), datetime.now()))

        file_name = j + '_' + i + '_' + start_time + '.csv'
        ohlc_data.to_csv(file_name)


mt.close()
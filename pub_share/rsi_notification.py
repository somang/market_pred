from binance.client import Client
import numpy as np
import pandas as pd
import smtplib, ssl
import time
import yaml
import btalib

CONFIG = yaml.load(open('./CONFIG.yml'))

API_KEY = CONFIG['binance_api']['key']
API_SECRET = CONFIG['binance_api']['secret']
user = CONFIG['gmail']['user']
password = CONFIG['gmail']['password']

print(API_KEY, API_SECRET, user, password)

client = Client(API_KEY, API_SECRET)
mcaps = {
    'small': ['BEL', 'BZRX', 'FIO', 'TRB', 'HARD', 'CTK', 'UNFI', 'LIT', 'PNT', 'SNT', 'RAMP', 'STORJ'],
    'med': ['ARK', 'AION', 'AERGO', 'DIA', 'WAVES', 'MKR', 'MANA', 'KNC', 'FTT', 'LSK'],
    'large': ['ADA', 'BCH', 'EOS', 'OMG', 'XLM', 'BAND', 'ALGO', 'XMR', 'SRM', 'ATOM', 'HBAR', 'ZEC', 'CRV', 'VET', 'CAKE', 'SOL', 'BNB', 'ETH', 'MATIC']
}

SYMBOLS = []
for cap in mcaps:
    SYMBOLS += mcaps[cap]

RSI_N = 14
RSI_THRESHOLD = 20
RUN_INTERVAL_MINS = 2

time_res = client.get_server_time()

def send_email(rsi_values, message):
    if len(rsi_values) > 0:
        message = "Subject: coin alert\n" + message  
        smtp_server = "smtp.gmail.com"
        sender_email = user + "@gmail.com"

        port = 465  # For SSL
        # Create a secure SSL context
        context = ssl.create_default_context()

        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, sender_email, message)
        
        print('email sent')

while True:
    rsi_values = []
    for SYMBOL in SYMBOLS:
        tok = SYMBOL+'BTC'
        klines = client.get_historical_klines(
            tok, Client.KLINE_INTERVAL_30MINUTE, '{} hrs ago UTC'.format((RSI_N+3)//2)
            # tok, Client.KLINE_INTERVAL_1HOUR, '{} hrs ago UTC'.format((RSI_N+3)//2)
        )
        
        for line in klines:
            del line[5:]
        coin_df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close'])
        coin_df.set_index('time', inplace=True)
        coin_df.index = pd.to_datetime(coin_df.index, unit='ms')
        coin_df["close"] = coin_df.close.astype(float)
        rsi = btalib.rsi(coin_df, period=RSI_N)
        rsi_val = rsi.df.rsi[-1]
        # print(tok, rsi_val)
        rsi_values.append((tok, rsi_val))

    rsi_values = list(filter(lambda x: x[1] < RSI_THRESHOLD, rsi_values))
    message = '\n'.join('{} {:.2f}'.format(s, r) for (s, r) in rsi_values)    
    print(message)

    # send_email(rsi_values, message)
    print('sleeping now...')
    time.sleep(60*RUN_INTERVAL_MINS)

import pandas as pd
import time
import datetime
import numpy as np
import csv

import pprint 
import twstock

pp = pprint.PrettyPrinter(indent=4)

# CRAWLING THE DATA OF STOCK, NO.2301

stock = twstock.Stock('2301')
print('get stock number')

for i in range(1995,2000):
    file_ = open('stock_data/stock_price_direction_'+str(i)+'.csv', 'w')
    csvCursor = csv.writer(file_)

    csvHeader = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'day_diff', 'up_or_down','direction']

    csvCursor.writerow(csvHeader)
    
    for x in range(1,13):
        print(i,x)
        one_company_stock = stock.fetch(i, x)
        for row in one_company_stock:
            temp = []
            if row[3] > row[6]:
                di = "u10"
            else:
                di = "u01"

            a = (row[4]-row[5])
            if a == 0.0:
                a = 1
            else:
                a = (row[3]-row[6])/a
            temp.extend((row[0],row[3],row[4],row[5],row[6],row[1],a,(row[3]-row[6]),di))
            csvCursor.writerow(temp)
        time.sleep(2) 

    time.sleep(2) 


file_ = open('stock_price_direction_2019.csv', 'w')
csvCursor = csv.writer(file_)

csvHeader = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'day_diff', 'up_or_down','direction']

csvCursor.writerow(csvHeader)

one_company_stock = stock.fetch(2018, 1)
for row in one_company_stock:
    temp = []
    if row[3] > row[6]:
        di = "u10"
    else:
        di = "u01"

    temp.extend((row[0],row[3],row[4],row[5],row[6],row[1],((row[3]-row[6])/(row[4]-row[5])),(row[3]-row[6]),di))
    csvCursor.writerow(temp)


# --------------------------------------------------------------------------------------------------------------

# CRAWLING THE DATA OF BITCOIN

# get market info for bitcoin from the start of 2016 to the current day
market_info = pd.read_html("https://coinmarketcap.com/currencies/bitcoin/historical-data/?start=20130428&end="+time.strftime("%Y%m%d"))[0]
market_info = market_info.assign(Date=pd.to_datetime(market_info['Date']))
# when Volume is equal to '-' convert it to 0
market_info.loc[market_info['Volume']=="-",'Volume']=0
market_info['Volume'] = market_info['Volume'].astype('int64')


# only take the record after 20170101
market_info = market_info[market_info['Date']>='2017-04-01']

# add some factor(attributes)
kwargs = { 'day_diff': lambda x: (x['Open']-x['Close'])/(x['High']-x['Low']),
            'up_or_down': lambda x: (x['Open']-x['Close']) }
market_info = market_info.assign(**kwargs)


df = pd.DataFrame(market_info)
df.to_csv("bit_price_direction.csv")

file = open('bit_price_direction.csv', 'r')
csvCursor_r = csv.reader(file)



file_ = open('bit_price_direction_.csv', 'w')
csvCursor = csv.writer(file_)

csvHeader = ['', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Market Cap', 'day_diff', 'up_or_down','direction']
# csvHeader = ['', 'Date', 'Volume', 'Market Cap', 'close_off_high', 'day_diff', 'volatility','direction']

csvCursor.writerow(csvHeader)

a = True
for row in csvCursor_r:
    if a:
        a = False
    else:
        if row[2] > row[5]:
            row.append("u100")
        elif row[2] == row[5]:
            row.append("u010")
        else:
            row.append("u001")
        csvCursor.writerow(row)



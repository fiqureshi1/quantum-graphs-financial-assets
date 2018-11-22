
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import pandas as pd

pd.set_option('display.max_columns', 30) 
pd.set_option('display.max_colwidth', -1) 

from time import time
from datetime import datetime
from itertools import product

import scipy.io as sio

import os
cwd = os.path.abspath(os.path.curdir) 

#----------------------------------------
def load_data(full_data, format):
#----------------------------------------

    t1 = time()
    print ('Starting Data Load...')
    
    if full_data:
        datafile = 'crypto-markets.csv' 
    else:
        datafile = 'crypto-markets.csv'
    
    
    df=pd.read_csv(datafile, sep=',') 
       
    print ('Data Loaded:', df.shape, 'in', round(time()-t1,4), 'seconds')
    print ("-----------------------------------------")
        
    if format == 'pd':
        return df.iloc[:,:-1]       # drop isFlaggedFraud
    elif format == 'np':
        return df.iloc[:,:-1].as_matrix()    # drop isFlaggedFraud
    else:
        print ("Warning: No format specified to LoadData()")
#----------------------------------------
# end load_data()
#----------------------------------------

#----------------------------------------
def processData(df):
#----------------------------------------
    print (df.columns)
    print ("-----------------------------------------")
    
    start_date = '2016-06-03'
    min_rank = 100
    
    # get the top ranking symbols which have been around since start_date
    slug_starts = df.loc[df.ranknow<min_rank].groupby(['slug'])['date'].agg('min')
    slug_ends = df.loc[df.ranknow<min_rank].groupby(['slug'])['date'].agg('max')
    slug_summary = pd.concat([slug_starts,slug_ends], axis=1, join='outer') 
    slug_summary.reset_index(inplace=True)  
    slug_summary.columns = ['slug', 'start', 'end']
    slug_list = slug_summary.loc[slug_summary.start<start_date].slug.tolist()
    
    # get symbol, date, close (price) for selected symbols
    data1 = df.loc[(df['slug'].isin(slug_list)) & (df['date']>start_date)]      
    data = data1.loc[:,['symbol','date','close']]
    data.reset_index(inplace=True, drop=True)
    
    
    # get daily price change (%)
    data['pct_ch'] = data.groupby('symbol')['close'].pct_change() + 1
    data = data.dropna()
    
    # get log return of daily price change (%)
    data['log_return']=np.log(data['pct_ch'])
    #print(data)
    
    prices=data
    
    symbol_list = []
    first_df = True
    for i, symb in enumerate (data.symbol.unique()):
        if symb != 'REP':
            symbol_list.append(symb)
            df_i = []
            df_i = data.loc[data.symbol==symb][['date','log_return']]
            df_i.columns = ['date',str(symb)]
            df_i.set_index('date', inplace=True, drop=True)
            df_i.index = pd.to_datetime(df_i.index)
            df_i2 = df_i.rolling(window=30, center=False).mean()
            if first_df:
                df_log_returns = df_i2
                first_df = False
            else:
                df_log_returns = pd.concat([df_log_returns,df_i2], axis=1, join='inner')
                
            #print ("SYMBO", symb, len(df_i))
    print ("selected {} symbols {}".format(len(symbol_list),symbol_list))
    print ("# days", len(df_log_returns))
    
    list18 = product([2018],[1,2,3,4,5])
    list17 = product([2017],[1,2,3,4,5,6,7,8,9,10,11,12])
    list16 = product([2016],[7,8,9,10,11,12])
    
    '''      
    month_list = list(list18)+list(list17)+list(list16)
    for j in (month_list):
        year = j[0]
        month = j[1] 
        df_month = df_log_returns.loc[(df_log_returns.index.year==year)&(df_log_returns.index.month==month)]
        df_month.to_csv('CryptoReturns_'+str(year)+'_'+str(month)+'.csv', header=False, index=False) 
        #if year == 2016 and month==7:
        #    print (j,df_month.head())

    pd.DataFrame(month_list).to_csv('CryptoMonths.csv', header=False, index=False)         
    pd.DataFrame(symbol_list).to_csv('CryptoSymbols.csv', header=True, index=False) 
    '''
    df_log_returns.to_csv('CryptoData.csv', header=True, index=True)  
    #print (df_log_returns.columns)
    #print (symbol_list)        
    print ("-----------------------------------------")
    return prices
    
#----------------------------------------
# end exploreDataAnalysis()
#----------------------------------------


#----------------------------------------
def main():
#----------------------------------------
    print ("-----------------------------------------")
    df = load_data(full_data=True, format='pd')
    prices = processData(df)

    prices["max"] = prices.groupby("symbol")["close"].transform(max)
    prices["min"] = prices.groupby("symbol")["close"].transform(min)
    prices["scaled_price"] = (prices["close"]-prices["min"]) / (prices["max"]-prices["min"])
    avg_crypto_prices = prices.groupby("date")["scaled_price"].mean()
    #print(avg_crypto_prices.columns)
    avg_crypto_prices.plot()
    avg_crypto_prices.to_csv('AvgCryptoPrices.csv', header=True, index=True)
    
    avg_crypto_prc_change=prices.groupby("date")["pct_ch"].mean() #.rolling(window=5, center=False).mean()
    avg_crypto_prc_change.plot()
    avg_crypto_prc_change.to_csv('AvgCryptoPriceChanges.csv', header=True, index=True)
    
    '''
    pd.DataFrame(symbol_list)
    P_array = sio.loadmat('P_mat.mat').get('P_mat')
    S_array = sio.loadmat('A_mat.mat').get('A_mat')
    A_array = sio.loadmat('S_mat.mat').get('S_mat')
    '''
    

#----------------------------------------
#end main()
#----------------------------------------
    
if __name__ == '__main__':
    main()
    

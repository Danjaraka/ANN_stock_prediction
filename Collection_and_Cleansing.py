#!/usr/bin/env python
# coding: utf-8
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import simfin as sf
#Used for column names
from simfin.names import *

#SIMFIN params
# Where all the csvs are stored
sf.set_data_dir('./simfin_data/')
# Simfin needs an api key (Using simfins free key so only get stock info from before 2018 )
sf.load_api_key(path='~/simfin_api_key.txt', default_key='free')
#offset the fundamental data by 50 days, the simfin documentation alsorecommends doing this..
dateOffset = pd.DateOffset(days=50)
refresh_days = 25
refresh_days_shareprices = 14
#Data collection from Simfin
hub = sf.StockHub(market='us', offset=dateOffset,refresh_days=refresh_days,refresh_days_shareprices=refresh_days_shareprices)
#Creating a panda dataframe with the simfin data
growthSignalsDf = hub.growth_signals(variant='daily')
valueSignalsDf = hub.val_signals(variant='daily')
financialSignalsDf = hub.fin_signals(variant='daily')
#Combining the 3 data frames into one big data frame
dfs = [financialSignalsDf, growthSignalsDf, valueSignalsDf]
signalsDf = pd.concat(dfs, axis=1)
#Drop the rows where all elements are missing.
signalsDf.dropna(how='all').head() 
df = signalsDf.dropna(how='all').reset_index(drop=True)
#Columns must have atleast 80% non NULL values, any that don't are dropped
#(Scikit cannot work well with lots of missing data)
thresh = 0.80 * len(signalsDf.dropna(how='all'))
signalsDf = signalsDf.dropna(axis='columns', thresh=thresh)
signalsDf.dropna(how='all').head()
# Name of the new column for the returns.
#This is the column the AI will attempt to predict
TOTAL_RETURN_1_3Y = 'Total Log Return 1-3 Years'
# Calculate the mean log-returns for all 1-3 year periods.
df_returns_1_3y = hub.mean_log_returns(name=TOTAL_RETURN_1_3Y,future=True, annualized=True,min_years=1, max_years=3)
#combine the two dataframes together
dfs = [signalsDf, df_returns_1_3y]
df_sig_rets = pd.concat(dfs, axis=1)
# Remove data outliers by winsorizing both the original stock data and the the Total return column
df_sig_rets = sf.winsorize(df_sig_rets)
# Remove all rows with any missing values
df_sig_rets = df_sig_rets.dropna(how='any')
# Remove all Stocks which have less than 150 data-rows.
df_sig_rets = df_sig_rets.groupby(TICKER).filter(lambda df: len(df)>150)
#Originally was saving to a csv but was having troubles keeping the dataframe original
#df_sig_rets.to_csv (r'/mnt/c/Users/danie/Documents/319 A2/simfin-tutorials/stockdata.csv', header=True)
#Serializes the data frame exactly as it is to storage
df_sig_rets.to_pickle("./stockdata.pkl")
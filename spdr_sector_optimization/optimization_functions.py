#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 13:58:05 2024

@author: paulobermann
# =============================================================================
# SPRD fund options are: 
# 1: 'XLB (Materials)'
# 2: 'XLC (Communication Services)'
# 3: 'XLE (Energy)'
# 4: 'XLF (Financials)'
# 5: 'XLI (Industrials)'
# 6: 'XLK (Technology)'
# 7: 'XLP (Consumer Staples)'
# 8: 'XLRE (Real Estate)'
# 9: 'XLU (Utilities)'
# 10: 'XLV (Healthcare)'
# 11: 'XLY (Consumer Discretionary)'
# =============================================================================
"""


#%%
import pandas as pd
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds


#%%
spdr_fund = 'XLP'
training_sdate = '2015-01'  ## Format: YYYY-MM
training_edate = '2015-12'  ## Format: YYYY-MM

testing_sdate = '2016-01'  # Format: YYYY-MM
testing_edate = '2016-12'  # Format: YYYY-MM

rf = 0.05  # Annual risk-free rate

startmoney = 10000


#%%
'''
Find the permnos for return data for the tickers that make up a certain index.

This should be the composition of that index as of the start date of the 
optimization period,
'''

# Find permnos for requested sector
sts = pd.read_csv('https://www.dropbox.com/scl/fi/p843qs28swgs1hj7l0q82/sectordecomp_ts-2025-04-05.csv?rlkey=dlm1l3pgj1ykzkb9vl19wrami&dl=1',
                      encoding='latin-1')
sts['date'] = pd.to_datetime(sts['datadate'])
date = pd.to_datetime(training_sdate)  # Convert Date to DateTime object
sub = sts.loc[sts['spdr_fund'] == spdr_fund, ].copy()  # only keep the wanted fund
sub['datediff'] = date - sub['date']


if len(sub.loc[sub['datediff']>=timedelta(0)]) != 0:
    sub = sub.loc[sub['datediff']>=timedelta(0)]  # this makes sure we don't use compositions from the future
    sub = sub.loc[sub['datediff'] == min(sub['datediff'])]
else: 
    sub = sub.loc[np.abs(sub['datediff']) == min(np.abs(sub['datediff']))]

print(f"Closest date: {sub.iloc[0]['date']}")
print(f'Requested date: {date}')
print(f"Date difference to requested date: {sub.iloc[0]['datediff']}")

permnos = sorted(set(sub['permno']))


#%% Print the sector as it is going to be used to optimze
printsub = sub[['datadate', 'index', 'spdr_fund', 'i', 'company']]
print(printsub)




#%%  Get the returns for the wanted permnos

returndf = pd.read_stata('/Users/paulobermann/Dropbox/Black Leaf Capital/data/CRSP_A_STOCK_MONTHLY - 2025-07-18.dta')
returndf.columns = [x.lower() for x in returndf.columns]
returndf = returndf[['permno', 'date', 'ret', 'comnam']]
returndf = returndf[returndf['permno'].isin(permnos)]
returndf.dropna(subset=['ret'], inplace=True)
returndf['month'] = returndf['date'].apply(lambda x: x.month)
returndf['year'] = returndf['date'].apply(lambda x: x.year)

for perm in permnos:
    if perm not in returndf['permno'].unique():
        print(f'{perm} not in return dataset')


### Subset to the training period ###
training_df = returndf.loc[
    (returndf['date']>=pd.to_datetime(training_sdate))
    & (returndf['date']<=pd.to_datetime(training_edate+'-31'))
    ].copy()

training_df.drop_duplicates(inplace=True)

# Make sure that all have a full set of data to train on
# Drop if there is 
nmonths_required = (pd.Period(training_edate, freq='M') - pd.Period(training_sdate, freq='M')).n + 1
training_df['count'] = training_df.groupby('permno')['permno'].transform('count')

counts = training_df[['permno', 'count']].drop_duplicates()
invalid_permnos = counts.loc[counts['count']!=nmonths_required, 'permno']


### Subsetting to the testing period ###
testing_df = returndf.loc[
    (returndf['date']>=pd.to_datetime(testing_sdate))
    & (returndf['date']<=pd.to_datetime(testing_edate+'-31'))
    ].copy()

testing_df.drop_duplicates(inplace=True)

nmonths_required_test = (pd.Period(testing_edate, freq='M') - pd.Period(testing_sdate, freq='M')).n + 1
testing_df['count'] = testing_df.groupby('permno')['permno'].transform('count')

testing_counts = testing_df[['permno', 'count']].drop_duplicates()
invalid_permnos_test = testing_counts.loc[testing_counts['count']!=nmonths_required_test, 'permno']


for p in invalid_permnos:
    print(f'{p} does not have required {nmonths_required} observations for training period. Will be dropped.')
for p in invalid_permnos_test:
    print(f'{p} does not have required {nmonths_required_test} observations for testing period. Will be dropped.')

training_df = training_df[~training_df['permno'].isin(invalid_permnos)]
training_df = training_df[~training_df['permno'].isin(invalid_permnos_test)]
training_df.reset_index(drop=True, inplace=True)
training_df = training_df.pivot(index='date', columns='permno', values='ret')

testing_df = testing_df[~testing_df['permno'].isin(invalid_permnos)]
testing_df = testing_df[~testing_df['permno'].isin(invalid_permnos_test)]
testing_df.reset_index(drop=True, inplace=True)
testing_df = testing_df.pivot(index='date', columns='permno', values='ret') 


#%%  Hashmap for mapping permnos to firm names and tickers
namemap = (returndf[['permno', 'comnam']]
           .drop_duplicates(subset='permno', keep='first')
           .set_index('permno')
           )


#%%  Get optimal portfolio weights
def getMaxSharpeWeights(returnData, rf, allowShorting=False):
    
    
    def calculateSharpeRatio(w, returnData, rf): 
        mu = returnData.mean()
        cov = returnData.cov()
        expReturn = w.dot(mu).item()
        var = w.dot(cov).dot(w.T).item()
        sd = np.sqrt(var)
    
        sharpeRatio = float((expReturn - rf/12) / (sd))
    
        return sharpeRatio


    def wrappedCalculateSharpeRatio(x):
        return(-calculateSharpeRatio(w=x, 
                                     returnData=rdata, 
                                     rf=rf))

    rdata = returnData
    n = rdata.shape[1]
    
    if allowShorting == False:
        # Non-negativity constraint
        bounds = Bounds([0] * n, [np.inf] * n)
    
    # Add to one constraint
    constraints = [
        {'type': 'eq',
         'fun': lambda x: np.sum(x) - 1},  # Sum of x's should be 1
    ]
    
    # Intial Guess: Equally-weighted
    x0 = np.full((1, n), 1/n).flatten()
    
    # Optimize
    results = minimize(wrappedCalculateSharpeRatio,
                       x0 = x0,
                       bounds = bounds, 
                       constraints = constraints)  
        
    series = pd.Series(results.x, index=rdata.columns)
    return series


wopt = getMaxSharpeWeights(returnData=training_df, rf=rf)
wopt = wopt.to_frame()

wopt = wopt.join(namemap, how='left').rename(columns={0: 'weight'})


#%%  Plot a bar chart of the optimal weights

nweights = len(wopt.loc[wopt['weight']>=0.01, 'weight'])
explode = [0.02]*nweights

plt.figure(figsize=(6,6), dpi=300)
plt.pie(wopt.loc[wopt['weight']>=0.01, 'weight'],
        labels=wopt.loc[wopt['weight']>=0.01, 'comnam'],
        autopct='%1.1f%%',
        explode=explode
        )
plt.title('Maximum Sharpe Ratio Portfolio')
plt.tight_layout()
plt.show()


#%% Calculate performance (Training Period)
# Calculate cumulative return for each asset, the calculate portfolio
initial_money = (wopt['weight'] * startmoney).rename('money')
cumulative_returns = (1+training_df).cumprod()
money_growth = initial_money * cumulative_returns
portoptmoney = money_growth.sum(axis=1).rename('port_money')

optportret = (portoptmoney.iloc[-1] / startmoney - 1)
print(f'Optimal Portfolio Return (Training Period): {optportret:.2%}')


# Calculate S&P 500 Performance for same period
sp50 = pd.read_excel('/Users/paulobermann/Dropbox/Black Leaf Capital/data/SP50_Prices.xlsx')
sp50 = sp50[['Date', 'Price']].set_index('Date')
sp50.sort_index(inplace=True)
sp50['return'] = sp50.pct_change()
sp50_train = sp50[training_sdate:training_edate].copy()
sp50_train['cumret'] = (1 + sp50_train['return']).cumprod()
sp50_train['money'] = startmoney * sp50_train['cumret']


sp50_series = sp50_train['money']
sp50ret = (sp50_train["money"].iloc[-1] / startmoney - 1)
print(f'S&P 500 Return (Training Period): {sp50ret:.2%}')


#%%  Calculate performance (Testing Period)
cumulative_returns_test = (1+testing_df).cumprod()
money_growth_test = initial_money * cumulative_returns_test
testportmoney = money_growth_test.sum(axis=1).rename('port_money')

testportreturn = testportmoney.iloc[-1] / startmoney - 1
print(f'Optimal Portfolio Return (Testing Period): {testportreturn:.2%}')


# Calculate S&P 500 Performance for the testing period
sp50_test = sp50[testing_sdate:testing_edate].copy()
sp50_test['cumret'] = (1 + sp50_test['return']).cumprod()
sp50_test['money'] = startmoney * sp50_test['cumret']


sp50_series_test = sp50_test['money']
sp50ret_test = (sp50_test["money"].iloc[-1] / startmoney - 1)
print(f'S&P 500 Return (Testing Period): {sp50ret_test:.2%}')


#%%  Plot Portfolio versus S&P 500 (Training Period)
plt.figure(figsize=(10,6), dpi=300)
plt.plot(portoptmoney, label='Optimal Portfolio')
plt.plot(sp50_series, label='S&P 500')
plt.suptitle(f'Optimal Portfolio Return vs. S&P 500 (Training Period)  ---  {training_sdate} to {training_edate}')
plt.title(f'Portfolio Return: {optportret:.2%} || S&P 500 Return: {sp50ret:.2%}')
plt.ylabel('Money')
plt.xticks(rotation=45)
plt.axhline(startmoney, linestyle='--', linewidth=0.5, color='black')
plt.legend()
plt.tight_layout()
plt.show()


#%%  Plot Portfolio versus S&P 500 (Testing Period)
plt.figure(figsize=(10,6), dpi=300)
plt.plot(testportmoney, label='Optimal Portfolio')
plt.plot(sp50_series_test, label='S&P 500')
plt.suptitle(f'Optimal Portfolio Return vs. S&P 500 (Testing Period)  ---  {testing_sdate} to {testing_edate}')
plt.title(f'Portfolio Return: {testportreturn:.2%} || S&P 500 Return: {sp50ret_test:.2%}')
plt.ylabel('Money')
plt.xticks(rotation=45)
plt.axhline(startmoney, linestyle='--', linewidth=0.5, color='black')
plt.legend()
plt.tight_layout()
plt.show()


#%%


# =============================================================================
# DEBUGGING
# =============================================================================







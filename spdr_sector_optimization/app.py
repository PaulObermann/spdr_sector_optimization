# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import timedelta
from scipy.optimize import minimize, Bounds

st.set_page_config(page_title="SPDR Optimal Portfolio Backtest", layout="wide")

st.title("SPDR Sector Optimal Portfolio Backtesting")

# Sidebar Inputs
with st.sidebar:
    st.header("Model Inputs")
    spdr_fund = st.selectbox(
        "Select SPDR Fund",
        ['XLB',
         'XLC',
         'XLE',
         'XLF',
         'XLI',
         'XLK',
         'XLP',
         'XLRE',
         'XLU',
         'XLV',
         'XLY'],
        index=6
    )
    training_sdate = st.text_input("Training Start Date (YYYY-MM)", value="2015-01")
    training_edate = st.text_input("Training End Date (YYYY-MM)", value="2015-12")
    testing_sdate = st.text_input("Testing Start Date (YYYY-MM)", value="2016-01")
    testing_edate = st.text_input("Testing End Date (YYYY-MM)", value="2016-12")
    rf = st.number_input("Annual Risk-Free Rate", value=0.05)
    startmoney = st.number_input("Starting Money", value=10000)

    run_button = st.button("Run Optimization")

if run_button:

    # Load sector decomposition
    @st.cache_data
    def load_sector_decomp():
        sts = pd.read_csv('https://www.dropbox.com/scl/fi/p843qs28swgs1hj7l0q82/sectordecomp_ts-2025-04-05.csv?rlkey=dlm1l3pgj1ykzkb9vl19wrami&dl=1', encoding='latin-1')
        sts['date'] = pd.to_datetime(sts['datadate'])
        return sts

    sts = load_sector_decomp()
    date = pd.to_datetime(training_sdate)
    sub = sts.loc[sts['spdr_fund'] == spdr_fund, ].copy()
    sub['datediff'] = date - sub['date']

    if len(sub.loc[sub['datediff']>=timedelta(0)]) != 0:
        sub = sub.loc[sub['datediff']>=timedelta(0)]
        sub = sub.loc[sub['datediff'] == min(sub['datediff'])]
    else:
        sub = sub.loc[np.abs(sub['datediff']) == min(np.abs(sub['datediff']))]

    st.write(f"Closest available composition date: **{sub.iloc[0]['date'].date()}** (Requested: {training_sdate})")
    st.write("### Sector Composition")
    st.dataframe(sub[['datadate', 'index', 'spdr_fund', 'i', 'company']])

    permnos = sorted(set(sub['permno']))

    # Load return data
    @st.cache_data
    def load_returns():
        returndf = pd.read_stata('/Users/paulobermann/Dropbox/Black Leaf Capital/data/CRSP_A_STOCK_MONTHLY - 2025-07-18.dta')
        returndf.columns = [x.lower() for x in returndf.columns]
        returndf = returndf[['permno', 'date', 'ret', 'comnam']]
        returndf['month'] = returndf['date'].apply(lambda x: x.month)
        returndf['year'] = returndf['date'].apply(lambda x: x.year)
        return returndf

    returndf = load_returns()
    returndf = returndf[returndf['permno'].isin(permnos)]
    returndf.dropna(subset=['ret'], inplace=True)

    # Training and testing subsets
    def prepare_df(df, sdate, edate):
        df_sub = df.loc[
            (df['date'] >= pd.to_datetime(sdate)) &
            (df['date'] <= pd.to_datetime(edate + '-31'))
        ].copy()
        df_sub.drop_duplicates(inplace=True)
        nmonths = (pd.Period(edate, freq='M') - pd.Period(sdate, freq='M')).n + 1
        df_sub['count'] = df_sub.groupby('permno')['permno'].transform('count')
        counts = df_sub[['permno', 'count']].drop_duplicates()
        invalid = counts.loc[counts['count'] != nmonths, 'permno']
        df_sub = df_sub[~df_sub['permno'].isin(invalid)]
        df_sub.reset_index(drop=True, inplace=True)
        return df_sub.pivot(index='date', columns='permno', values='ret')

    training_df = prepare_df(returndf, training_sdate, training_edate)
    testing_df = prepare_df(returndf, testing_sdate, testing_edate)

    namemap = returndf[['permno', 'comnam']].drop_duplicates(subset='permno').set_index('permno')

    # Optimization
    def getMaxSharpeWeights(returnData, rf, allowShorting=False):
        def calculateSharpeRatio(w, returnData, rf):
            mu = returnData.mean()
            cov = returnData.cov()
            expReturn = w.dot(mu).item()
            var = w.dot(cov).dot(w.T).item()
            sd = np.sqrt(var)
            return float((expReturn - rf/12) / sd)

        def wrappedCalculateSharpeRatio(x):
            return -calculateSharpeRatio(x, returnData, rf)

        n = returnData.shape[1]
        bounds = Bounds([0]*n, [np.inf]*n) if not allowShorting else None
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]
        x0 = np.full(n, 1/n)

        results = minimize(wrappedCalculateSharpeRatio, x0, bounds=bounds, constraints=constraints)
        return pd.Series(results.x, index=returnData.columns)

    wopt = getMaxSharpeWeights(training_df, rf)
    wopt = wopt.to_frame(name="weight").join(namemap)

    # Pie chart
    st.subheader("Optimal Portfolio Weights")
    fig1, ax1 = plt.subplots(figsize=(5,5))
    wplot = wopt[wopt['weight']>=0.01]
    ax1.pie(wplot['weight'], labels=wplot['comnam'], autopct='%1.1f%%', explode=[0.02]*len(wplot))
    ax1.set_title('Maximum Sharpe Ratio Portfolio')
    st.pyplot(fig1)

    # Performance calculations
    def calc_perf(df, wopt, startmoney):
        initial_money = (wopt['weight'] * startmoney).rename('money')
        cumulative_returns = (1+df).cumprod()
        money_growth = initial_money * cumulative_returns
        portmoney = money_growth.sum(axis=1).rename('port_money')
        return portmoney, (portmoney.iloc[-1] / startmoney - 1)

    portoptmoney, optportret = calc_perf(training_df, wopt, startmoney)
    testportmoney, testportreturn = calc_perf(testing_df, wopt, startmoney)

    # Load S&P 500
    @st.cache_data
    def load_sp500():
        sp50 = pd.read_excel('/Users/paulobermann/Dropbox/Black Leaf Capital/data/SP50_Prices.xlsx')
        sp50 = sp50[['Date', 'Price']].set_index('Date')
        sp50.sort_index(inplace=True)
        sp50['return'] = sp50.pct_change()
        return sp50

    sp50 = load_sp500()

    def sp500_perf(sp50, sdate, edate):
        sp = sp50[sdate:edate].copy()
        sp['cumret'] = (1 + sp['return']).cumprod()
        sp['money'] = startmoney * sp['cumret']
        return sp['money'], (sp["money"].iloc[-1] / startmoney - 1)

    sp50_series, sp50ret = sp500_perf(sp50, training_sdate, training_edate)
    sp50_series_test, sp50ret_test = sp500_perf(sp50, testing_sdate, testing_edate)

    # Training plot
    st.subheader("Training Period Performance")
    fig2, ax2 = plt.subplots(figsize=(6,4))
    ax2.plot(portoptmoney, label='Optimal Portfolio')
    ax2.plot(sp50_series, label='S&P 500')
    ax2.axhline(startmoney, linestyle='--', color='black', linewidth=0.5)
    ax2.set_title(f'Training Portfolio: {optportret:.2%} | S&P 500: {sp50ret:.2%}')
    ax2.set_ylabel("Money")
    ax2.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig2)

    # Testing plot
    st.subheader("Testing Period Performance")
    fig3, ax3 = plt.subplots(figsize=(6,4))
    ax3.plot(testportmoney, label='Optimal Portfolio')
    ax3.plot(sp50_series_test, label='S&P 500')
    ax3.axhline(startmoney, linestyle='--', color='black', linewidth=0.5)
    ax3.set_title(f'Testing Portfolio: {testportreturn:.2%} | S&P 500: {sp50ret_test:.2%}')
    ax3.set_ylabel("Money")
    ax3.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig3)

    # Summary
    st.write(f"**Training Return:** {optportret:.2%} vs. S&P 500: {sp50ret:.2%}")
    st.write(f"**Testing Return:** {testportreturn:.2%} vs. S&P 500: {sp50ret_test:.2%}")

else:
    st.info("Configure inputs in the sidebar and click **Run Optimization** to begin.")

import pandas as pd
import numpy as np
import os
import datetime as dt

import torch

def compute_monthly_cols(symbol_df):
    returns = symbol_df.pct_change()
    # prev_12_returns = symbol_df.Close.pct_change(12) + 1
    # prev_6_returns = symbol_df.Close.pct_change(6) + 1
    # prev_3_returns = symbol_df.Close.pct_change(3) + 1

    # rolling_12 = symbol_df.Close.rolling(window=12)
    # rolling_6 = symbol_df.Close.rolling(window=6)
    # rolling_3 = symbol_df.Close.rolling(window=3)
    # rolling_2 = symbol_df.Close.rolling(window=2)

    # rolling_returns = returns.rolling(12)

    prev_365_returns = symbol_df.pct_change(365)
    prev_120_returns = symbol_df.pct_change(120)
    prev_30_returns = symbol_df.pct_change(30)
    prev_7_returns = symbol_df.pct_change(7)
    prev_3_returns = symbol_df.pct_change(3)

    rolling_365 = symbol_df.rolling(window=365)
    rolling_120 = symbol_df.rolling(window=120)
    rolling_30 = symbol_df.rolling(window=30)
    rolling_7 = symbol_df.rolling(window=7)
    rolling_3 = symbol_df.rolling(window=3)

    rolling_returns = returns.rolling(7)


    result_data = {
        "next19_return": returns.shift(-19),
        "next18_return": returns.shift(-18),
        "next17_return": returns.shift(-17),
        "next16_return": returns.shift(-16),
        "next15_return": returns.shift(-15),
        "next14_return": returns.shift(-14),
        "next13_return": returns.shift(-13),
        "next12_return": returns.shift(-12),
        "next11_return": returns.shift(-11),
        "next10_return": returns.shift(-10),
        "next9_return": returns.shift(-9),
        "next8_return": returns.shift(-8),
        "next7_return": returns.shift(-7),
        "next6_return": returns.shift(-6),
        "next5_return": returns.shift(-5),
        "next4_return": returns.shift(-4),
        "next3_return": returns.shift(-3),
        "next2_return": returns.shift(-2),
        "next1_return": returns.shift(-1),
        "next19_cum_return": symbol_df.pct_change(19).shift(-19),
        "cur_return": returns,
        "prev1_return": returns.shift(1),
        "prev2_return": returns.shift(2),
        "prev3_return": returns.shift(3),
        "prev4_return": returns.shift(4),
        "prev5_return": returns.shift(5),
        "prev6_return": returns.shift(6),
        "prev7_return": returns.shift(7),
        "prev8_return": returns.shift(8),
        "prev9_return": returns.shift(9),
        "prev10_return": returns.shift(10),
        "prev19_cum_return": symbol_df.pct_change(19),
        "prev_year_return": prev_365_returns,
        "prev_qtr_return": prev_120_returns,
        "prev_month_returns": prev_30_returns,
        "prev_week_returns": prev_7_returns,

        "return_rolling_mean": rolling_returns.mean(),
        "return_rolling_var": rolling_returns.var(),
        #         "return_rolling_12_min": rolling_returns.min(),
        #         "return_rolling_12_max": rolling_returns.max(),

        "rolling_365_mean": rolling_365.mean(),
        "rolling_365_var": rolling_365.var(),

        "rolling_120_mean": rolling_120.mean(),
        "rolling_120_var": rolling_120.var(),

        "rolling_30_mean": rolling_30.mean(),
        "rolling_30_var": rolling_30.var(),
        #         "rolling_12_min": rolling_12.min(),
        #         "rolling_12_max": rolling_12.max(),

        "rolling_7_mean": rolling_7.mean(),
        "rolling_7_var": rolling_7.var(),
        #         "rolling_6_min": rolling_6.min(),
        #         "rolling_6_max": rolling_6.max(),

        "rolling_3_mean": rolling_3.mean(),
        "rolling_3_var": rolling_3.var(),
        #         "rolling_3_min": rolling_3.min(),
        #         "rolling_3_max": rolling_3.max(),

        #         "rolling_2_mean": rolling_2.mean(),
        #         "rolling_2_var": rolling_2.var(),
        #         "rolling_2_min": rolling_2.min(),
        #         "rolling_2_max": rolling_2.max(),

    }
    feature_data = pd.DataFrame(result_data)
    return feature_data

def feature_w_raw_price(symbol_df):
    returns = symbol_df.pct_change()
    # prev_12_returns = symbol_df.Close.pct_change(12) + 1
    # prev_6_returns = symbol_df.Close.pct_change(6) + 1
    # prev_3_returns = symbol_df.Close.pct_change(3) + 1

    # rolling_12 = symbol_df.Close.rolling(window=12)
    # rolling_6 = symbol_df.Close.rolling(window=6)
    # rolling_3 = symbol_df.Close.rolling(window=3)
    # rolling_2 = symbol_df.Close.rolling(window=2)

    # rolling_returns = returns.rolling(12)

    prev_365_returns = symbol_df.pct_change(365)
    prev_120_returns = symbol_df.pct_change(120)
    prev_30_returns = symbol_df.pct_change(30)
    prev_7_returns = symbol_df.pct_change(7)
    prev_3_returns = symbol_df.pct_change(3)

    rolling_365 = symbol_df.rolling(window=365)
    rolling_120 = symbol_df.rolling(window=120)
    rolling_30 = symbol_df.rolling(window=30)
    rolling_7 = symbol_df.rolling(window=7)
    rolling_3 = symbol_df.rolling(window=3)

    rolling_returns = returns.rolling(7)


    result_data = {
        "next19_return": returns.shift(-19),
        "next18_return": returns.shift(-18),
        "next17_return": returns.shift(-17),
        "next16_return": returns.shift(-16),
        "next15_return": returns.shift(-15),
        "next14_return": returns.shift(-14),
        "next13_return": returns.shift(-13),
        "next12_return": returns.shift(-12),
        "next11_return": returns.shift(-11),
        "next10_return": returns.shift(-10),
        "next9_return": returns.shift(-9),
        "next8_return": returns.shift(-8),
        "next7_return": returns.shift(-7),
        "next6_return": returns.shift(-6),
        "next5_return": returns.shift(-5),
        "next4_return": returns.shift(-4),
        "next3_return": returns.shift(-3),
        "next2_return": returns.shift(-2),
        "next1_return": returns.shift(-1),
        "next19_cum_return": symbol_df.pct_change(19).shift(-19),
        "raw_price" : symbol_df,
        "cur_return": returns,
        "prev1_return": returns.shift(1),
        "prev2_return": returns.shift(2),
        "prev3_return": returns.shift(3),
        "prev4_return": returns.shift(4),
        "prev5_return": returns.shift(5),
        "prev6_return": returns.shift(6),
        "prev7_return": returns.shift(7),
        "prev8_return": returns.shift(8),
        "prev9_return": returns.shift(9),
        "prev10_return": returns.shift(10),
        "prev19_cum_return": symbol_df.pct_change(19),
        "prev_year_return": prev_365_returns,
        "prev_qtr_return": prev_120_returns,
        "prev_month_returns": prev_30_returns,
        "prev_week_returns": prev_7_returns,

        "return_rolling_mean": rolling_returns.mean(),
        "return_rolling_var": rolling_returns.var(),
        #         "return_rolling_12_min": rolling_returns.min(),
        #         "return_rolling_12_max": rolling_returns.max(),

        "rolling_365_mean": rolling_365.mean(),
        "rolling_365_var": rolling_365.var(),

        "rolling_120_mean": rolling_120.mean(),
        "rolling_120_var": rolling_120.var(),

        "rolling_30_mean": rolling_30.mean(),
        "rolling_30_var": rolling_30.var(),
        #         "rolling_12_min": rolling_12.min(),
        #         "rolling_12_max": rolling_12.max(),

        "rolling_7_mean": rolling_7.mean(),
        "rolling_7_var": rolling_7.var(),
        #         "rolling_6_min": rolling_6.min(),
        #         "rolling_6_max": rolling_6.max(),

        "rolling_3_mean": rolling_3.mean(),
        "rolling_3_var": rolling_3.var(),
        #         "rolling_3_min": rolling_3.min(),
        #         "rolling_3_max": rolling_3.max(),

        #         "rolling_2_mean": rolling_2.mean(),
        #         "rolling_2_var": rolling_2.var(),
        #         "rolling_2_min": rolling_2.min(),
        #         "rolling_2_max": rolling_2.max(),

    }
    feature_data = pd.DataFrame(result_data)
    return feature_data

def get_price_feature_matrix(price_feature_df):
    num_dates, num_assets = map(len, price_feature_df.index.levels)
    price_matrix = price_feature_df.values.reshape((num_dates, num_assets, -1))
    return price_matrix


def load_data(market,raw_include = 0):
    raw_price_df = pd.read_csv('data/{}/data.csv'.format(market))
    ## make feature _array (471,1010,37)
    for i in range(1,raw_price_df.shape[1]):
        if raw_include == 0:
            price_feature_df = compute_monthly_cols(raw_price_df.iloc[:,i])
        else:
            price_feature_df = feature_w_raw_price(raw_price_df.iloc[:,i])
        price_feature_df.index = raw_price_df.Date
        price_feature_df.dropna(inplace=True)
        if i == 1:
            feature_array = np.expand_dims(np.array(price_feature_df), axis=0)
        else:
            feature_array1 = np.expand_dims(np.array(price_feature_df), axis=0)
            feature_array= np.vstack([feature_array, feature_array1])
    
    ## make multiindex         
    prod = [list(price_feature_df.index), list(raw_price_df.columns)[1:]]
    indexs = pd.MultiIndex.from_product(prod, names = ['Date', 'Symbol'])
    indexs
    
    ## Adapt multiindex for feature array
    feat_to_df = []
    for i in range(len(list(price_feature_df.index))):
        for j in range(len(list(raw_price_df.columns)[1:])):
            feat_to_df.append(feature_array[j,i,:])

    price_feature_df = pd.DataFrame(feat_to_df, index= indexs, columns= price_feature_df.columns)

    
    target_names = ['next19_cum_return'] ## 한달 동안의 누적 수익률(예측해야할 값)
    hist_names = ["prev19_cum_return"]
    future_names = ["next{}_return".format(i) for i in range(1,20)]
    feature_cols = [c for c in price_feature_df.columns if c not in target_names + hist_names +future_names]
    target_mat = torch.tensor(get_price_feature_matrix(price_feature_df[target_names]))
    future_mat = torch.tensor(get_price_feature_matrix(price_feature_df[hist_names+future_names]))
    feature_mat = torch.tensor(get_price_feature_matrix(price_feature_df[feature_cols]))
    dates = list(price_feature_df.index.levels[0])
    symbols = list(price_feature_df.index.levels[1])
    
    return feature_mat, target_mat, feature_cols, future_mat, target_names, dates, symbols


def load_target_index():
    index_df = pd.read_csv('data/sp500.csv')
    index_df['Date'] = index_df['Date'].apply(lambda x: pd.to_datetime(x, format= "%Y-%m-%d"))
    index_df['next19_cum_return'] = index_df['^GSPC'].pct_change(19).shift(-19)
    index_df = index_df.loc[(index_df['Date']>='2019-12-12') & (index_df['Date']<='2023-12-01')]
    index_df.set_index(keys='Date', inplace=True)
    index_df = index_df['next19_cum_return']
    target_mat = np.array(index_df).reshape(-1, 1,1)
    
    return torch.tensor(target_mat)

def load_prev_marketcap(dates,market):
    cap_df = pd.read_csv('data/{}/cap_weight.csv'.format(market))
    cap_df['Date'] = cap_df['Date'].apply(lambda x: pd.to_datetime(x, format= "%Y-%m-%d"))
    cap_df.set_index(keys='Date', inplace=True)
    # cap_df = cap_df.shift(-1).loc[dates,:]
    cap_df = cap_df.loc[dates,:]
    target_mat = np.array(cap_df)
    
    return torch.tensor(target_mat)

def load_bench_data():
    raw_price_df = pd.read_csv('data/data.csv')
    raw_price_df['Date'] = raw_price_df['Date'].apply(lambda x : pd.to_datetime(x, format = "%Y-%m-%d"))
    raw_price_df.iloc[:,1:] = raw_price_df.iloc[:,1:].pct_change()
    raw_price_df = raw_price_df.loc[(raw_price_df['Date']>='2019-12-12')&(raw_price_df['Date']<='2023-12-01')]
    raw_price_df.set_index(keys='Date', inplace=True)
    target_mat = np.array(raw_price_df).reshape(-1, 511)
    return torch.tensor(target_mat)

def load_bench_target_index():
    index_df = pd.read_csv('data/sp500.csv')
    index_df['Date'] = index_df['Date'].apply(lambda x: pd.to_datetime(x, format= "%Y-%m-%d"))
    index_df['return'] = index_df['^GSPC'].pct_change()
    index_df = index_df.loc[(index_df['Date']>='2019-12-12') & (index_df['Date']<='2023-12-01')]
    index_df.set_index(keys='Date', inplace=True)
    index_df = index_df['return']
    target_mat = np.array(index_df).reshape(-1)
    
    return torch.tensor(target_mat)  

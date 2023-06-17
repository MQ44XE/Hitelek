import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import math

#5-year historical data
df1 = pd.read_csv("SPY.csv", index_col=0)
df1.columns = [colname+"_SPY" for colname in df1.columns]
df2 = pd.read_csv("AGG.csv", index_col=0)
df2.columns = [colname+"_AGG" for colname in df2.columns]

#adding returns to dataframe
df1["Daily_return_SPY"] = np.log(df1["Adj Close_SPY"] / df1["Adj Close_SPY"].shift(1))
df2["Daily_return_AGG"] = np.log(df2["Adj Close_AGG"] / df2["Adj Close_AGG"].shift(1))

#gathering returns into one dataframe
only_returns = pd.DataFrame()
only_returns["SPY"]=df1["Daily_return_SPY"]
#only_returns["AGG"]=df2["Daily_return_AGG"]
only_returns = only_returns.dropna()

def read_etf_file(etf):
    filename = os.path.join(etf + '.csv')
    df = pd.read_csv(filename, index_col=0)
    df.index = pd.to_datetime(df.index)
    return df


def get_etf_returns(etf_name,
    return_type='log', fieldname='Adj Close'):

    df = read_etf_file(etf_name)
    df = df[[fieldname]]

    df['shifted'] = df.shift(1)
    if return_type=='log':
        df['return'] = np.log(df[fieldname]/df['shifted'])
    if return_type=='simple':
        df['return'] = df[fieldname]/df['shifted']-1

    # restrict df to result col
    df = df[['return']]
    # rename column
    df.columns = [etf_name]
    # df = df.rename(by=col, {'return': etf_name})
    df.dropna(inplace=True)
    return df


def get_total_return(etf, return_type='log'):
    return get_etf_returns(etf, return_type, 'Adj Close')


def get_dividend_return(etf, return_type='log'):
    # 1 calc total simple return from Adj Close and Close
    df_ret_from_adj = get_etf_returns(etf, 'simple', 'Adj Close')
    df_ret_from_close = get_etf_returns(etf, 'simple', 'Close')
    # 2 simple div = ret Adj Close simple - ret Close simple
    df_div = df_ret_from_adj - df_ret_from_close
    # 3 convert to log if log
    if return_type=='log':
        df_div = np.log(df_div + 1)
    return df_div


def get_price_return(etf, return_type='log'):
    df_total = get_total_return(etf, 'simple')
    df_div = get_dividend_return(etf, 'simple')
    df_price = df_total - df_div
    if return_type == 'log':
        df_price = np.log(df_price + 1)
    return df_price

def get_joined_returns(d_weights, from_date=None, to_date=None):
    l_df = []
    for etf, value in d_weights.items():
        df_temp = get_total_return(etf, return_type='simple')
        l_df.append(df_temp)
    df_joined = pd.concat(l_df, axis=1)
    df_joined.sort_index(inplace=True)
    df_joined.dropna(inplace=True)
    fromdate = pd.to_datetime(from_date)
    todate = pd.to_datetime(to_date)
    filtered_df = df_joined.loc[fromdate:todate]
    return filtered_df


def get_portfolio_returns(d_weights):
    l_df = []
    for etf, value in d_weights.items():
        df_temp = get_total_return(etf, return_type='simple')
        l_df.append(df_temp)
    df_joined = pd.concat(l_df, axis=1)
    df_joined.sort_index(inplace=True)
    df_joined.dropna(inplace=True)
    df_weighted_returns = df_joined * pd.Series(d_weights)
    s_portfolio_return = df_weighted_returns.sum(axis=1)
    return pd.DataFrame(s_portfolio_return, columns=['pf'])

def get_portfolio_return_btw_dates(d_weights,
    from_date=None, to_date=None):
    df = get_portfolio_returns(d_weights)
    fromdate = pd.to_datetime(from_date)
    todate = pd.to_datetime(to_date)
    filtered_df = df.loc[fromdate:todate]
    return filtered_df

def subtract_trading_date(actual_date, x):
    date = pd.to_datetime(actual_date)
    # create a date range from the current date to `x` days ago
    date_range = pd.bdate_range(end=date, periods=x + 1)
    # subtract the last date in the range from the current date
    result = date_range[0]
    result_str = result.strftime('%Y-%m-%d')
    return result_str

def calc_simple_var(pf_value, d_weights, l_conf_levels,
    last_day_of_interval, window_in_days):
    from_date = subtract_trading_date(last_day_of_interval, window_in_days)
    df_ret = get_portfolio_return_btw_dates(
        d_weights, from_date, last_day_of_interval)
    l_quantiles = [1 - x for x in l_conf_levels]
    pf_mean = float(df_ret.mean())
    pf_std = float(df_ret.std())
    var_numbers = norm.ppf(l_quantiles, loc=pf_mean, scale=pf_std)
    df_result_ret = pd.DataFrame(var_numbers)
    df_result_ret.index = l_conf_levels
    df_result_ret = df_result_ret.transpose()
    df_result_ret.index = [last_day_of_interval]
    df_result_amount = df_result_ret * pf_value
    return df_result_ret, df_result_amount

def calc_covar_var(pf_value, d_weights, l_conf_levels,
    last_day_of_interval, window_in_days):
    from_date = subtract_trading_date(last_day_of_interval, window_in_days)
    df_rets = get_joined_returns(
        d_weights, from_date, last_day_of_interval)
    l_quantiles = [1 - x for x in l_conf_levels]
    means = df_rets.mean()
    covar = df_rets.cov()
    s_weights = pd.Series(d_weights)
    pf_mean = (s_weights * means).sum()
    pf_var = np.dot(s_weights.T, np.dot(covar, s_weights))
    var_numbers = norm.ppf(l_quantiles, loc=pf_mean, scale=np.sqrt(pf_var))
    df_result_ret = pd.DataFrame(var_numbers)
    df_result_ret.index = l_conf_levels
    df_result_ret = df_result_ret.transpose()
    df_result_ret.index = [last_day_of_interval]
    df_result_amount = df_result_ret * pf_value
    return df_result_ret, df_result_amount

d_weights = {'SPY': 0.1, 'AGG': 0.9}
df_portfolio_returns = get_portfolio_returns(d_weights)

def calculate_historical_var(df_portfolio_returns, alpha):
    l_quantiles = 1-alpha
    df_pf = df_portfolio_returns
    df_result = df_pf.quantile(l_quantiles)
    return df_result

def calc_var_for_period(vartype,
    pf_value, d_weights, l_conf_levels,
    from_date, to_date,
    window_in_days):
    d_var_f = {
        'hist': calculate_historical_var,
        'simple': calc_simple_var,
        'covar': calc_covar_var
    }
    f_var = d_var_f[vartype]
    business_days = pd.date_range(start=from_date, end=to_date, freq='B')
    df_result = None
    for last_day_of_interval in business_days:
        df_temp_, df_temp_amount = f_var(
            pf_value, d_weights, l_conf_levels,
            last_day_of_interval, window_in_days)
        if df_result is None:
            df_result = df_temp_amount
        else:
            df_result = pd.concat(
                [df_result, df_temp_amount],
                axis=0)
    return df_result

def calc1_historical_var():
    d_weights = {'SPY': 0.1, 'AGG': 0.9}
    l_conf_levels = [0.95, 0.99]
    l_quantiles = [1 - x for x in l_conf_levels]
    df_pf = get_portfolio_returns(d_weights)
    df_result = df_pf.quantile(l_quantiles)
    df_result.index = l_conf_levels
    return df_result

def calc2_historical_var():
    d_weights = {'SPY': 0.3, 'AGG': 0.7}
    l_conf_levels = [0.95, 0.99]
    l_quantiles = [1 - x for x in l_conf_levels]
    df_pf = get_portfolio_returns(d_weights)
    df_result = df_pf.quantile(l_quantiles)
    df_result.index = l_conf_levels
    return df_result

def calc3_historical_var():
    d_weights = {'SPY': 0.5, 'AGG': 0.5}
    l_conf_levels = [0.95, 0.99]
    l_quantiles = [1 - x for x in l_conf_levels]
    df_pf = get_portfolio_returns(d_weights)
    df_result = df_pf.quantile(l_quantiles)
    df_result.index = l_conf_levels
    return df_result

def calc4_historical_var():
    d_weights = {'SPY': 0.7, 'AGG': 0.3}
    l_conf_levels = [0.95, 0.99]
    l_quantiles = [1 - x for x in l_conf_levels]
    df_pf = get_portfolio_returns(d_weights)
    df_result = df_pf.quantile(l_quantiles)
    df_result.index = l_conf_levels
    return df_result

def calc5_historical_var():
    d_weights = {'SPY': 0.9, 'AGG': 0.1}
    l_conf_levels = [0.95, 0.99]
    l_quantiles = [1 - x for x in l_conf_levels]
    df_pf = get_portfolio_returns(d_weights)
    df_result = df_pf.quantile(l_quantiles)
    df_result.index = l_conf_levels
    return df_result


historical_var=pd.DataFrame()
historical_var["SPY_0.1-AGG_0.9"] = calc1_historical_var()
historical_var["SPY_0.3-AGG_0.7"] = calc2_historical_var()
historical_var["SPY_0.5-AGG_0.5"] = calc3_historical_var()
historical_var["SPY_0.7-AGG_0.3"] = calc4_historical_var()
historical_var["SPY_0.9-AGG_0.1"] = calc5_historical_var()

# print(historical_var)
# print(get_portfolio_returns({'SPY': 0.1, 'AGG': 0.9}))
# print(calculate_historical_var(df_portfolio_returns,0.99))
# print(calc1_historical_var())

#Exercise 2
def calculate_historical_volatility(df_returns, annualized=False):
    volatility = np.std(df_returns)
    if annualized == True:
        volatility = volatility * np.sqrt(252)  #252 napos kereskedéssel számolva egy évben
    return volatility

def potential_future_price_path(df_returns, volatility, length, simulations=1):
    mean = np.mean(df_returns)
    future_returns = np.random.normal(mean, volatility, (length, simulations))
    return future_returns

d_weights = {'SPY': 0.1, 'AGG': 0.9}
def risk_adjusted_weights(etf_names):
    volatilities = []
    for etf in etf_names:
        volatilities.append(calculate_historical_volatility(get_etf_returns(etf)))
    weights = []
    base = np.sum(volatilities)
    for vol in volatilities:
        weights.append(1-vol/base)
    return weights

def simulated_returns(expected_return, volatility, correlation, numOfSim):
    cov = np.array([[volatility[0] ** 2, correlation * volatility[0] * volatility[1]],
                           [correlation * volatility[0] * volatility[1], volatility[1] ** 2]])
    future_returns = np.random.multivariate_normal(expected_return, cov, numOfSim).T
    weights = risk_adjusted_weights(["SPY", "AGG"])
    portfolio_returns = future_returns[0]*float(weights[0]) + future_returns[1]*float(weights[1])
    return portfolio_returns

etf = ["SPY","AGG"]
expected_return = []
volatility = []
for x in etf:
    expected_return.append(get_etf_returns(x))
    volatility.append(calculate_historical_volatility(get_etf_returns(x)))
expected_return = np.mean(expected_return, axis=1)
expected_return = [expected_return[0][0], expected_return[1][0]]
volatility = np.array([volatility[0], volatility[1]])
volatility = [volatility[0][0], volatility[1][0]]

corr_1 = pd.DataFrame(simulated_returns(expected_return, volatility, 0.8, 10000), columns = ['Portfolio corr=0.8'])
corr_2 = pd.DataFrame(simulated_returns(expected_return, volatility, 0.2, 10000), columns = ['Portfolio corr=0.2'])
corr_3 = pd.DataFrame(simulated_returns(expected_return, volatility, -0.2, 10000), columns = ['Portfolio corr=-0.2'])
corr_4 = pd.DataFrame(simulated_returns(expected_return, volatility, -0.8, 10000), columns = ['Portfolio corr=-0.8'])

print(calculate_historical_var(corr_1, 0.95))
print(calculate_historical_var(corr_2, 0.95))
print(calculate_historical_var(corr_3, 0.95))
print(calculate_historical_var(corr_4, 0.95))


# Exercise 3
def calculate_ewma_variance(df_etf_returns, decay_factor, window):
    squared_returns = df_etf_returns ** 2
    ewma_var = squared_returns.ewm(alpha=1-decay_factor, min_periods=window, adjust=True).mean()
    return ewma_var

# Calculate EWMA variance with decay factor 0.94 and window of 100 days
ewma_var_1 = calculate_ewma_variance(only_returns, 0.94, 100)

# Calculate EWMA variance with decay factor 0.97 and window of 100 days
ewma_var_2 = calculate_ewma_variance(only_returns, 0.97, 100)

# Plot the EWMA variances
plt.figure(figsize=(12, 6))
plt.plot(ewma_var_1, label='EWMA Variance (Decay 0.94)')
plt.plot(ewma_var_2, label='EWMA Variance (Decay 0.97)')
plt.xlabel('Date')
plt.ylabel('Variance')
plt.title('EWMA Variance of ETF Returns')
plt.legend()
plt.show()
print(ewma_var_1)
print(ewma_var_2)



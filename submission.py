import pandas as pd
import numpy as np

#5-year historical data
df1 = pd.read_csv("SPY.csv", index_col=0)
df1.columns = [colname+"_SPY" for colname in df1.columns]
df2 = pd.read_csv("AGG.csv", index_col=0)
df2.columns = [colname+"_AGG" for colname in df2.columns]

#adding returns to dataframe
df1["Daily_return_SPY"] = df1["Adj Close_SPY"] / df1["Adj Close_SPY"].shift(1) - 1
df2["Daily_return_AGG"] = df2["Adj Close_AGG"] / df2["Adj Close_AGG"].shift(1) - 1

#gathering returns into one dataframe
only_returns = pd.DataFrame()
only_returns["SPY"]=df1["Daily_return_SPY"]
only_returns["AGG"]=df2["Daily_return_AGG"]
from cProfile import label
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

# creating objectives/trading conditions that we want to predict
def create_classification_trading_condition(df):
    df['open-close'] = df['open'] - df['close']
    df['high-low'] = df['high'] - df['low']
    df = df.dropna()
    X = df[['open-close', 'high-low']]
    Y = np.where(df['close'].shift(-1) > df['close'], 1, -1)
    return (X, Y)

def create_regression_trading_condition(df):
    df['open-close'] = df['open'] - df['close']
    df['high-low'] = df['high'] - df['low']
    df['target'] = df['close'].shift(-1) - df['close']
    df = df.dropna()
    X = df[['open-close', 'high-low']]
    Y = df[['target']]
    return (df, X, Y)

# partitioning datasets into training and testing datasets
def create_train_split_group(X, Y, split_ratio=0.8):
    return train_test_split(X, Y, shuffle=False, train_size=split_ratio)

stock_data = pd.read_csv('zhgf.csv', index_col=0)

# liner regression methods 线性回归

# Ordinary Least Squares
# print(stock_data)
stock_data, X, Y = create_regression_trading_condition(stock_data)
print(stock_data)
# print(X)
# print(Y)
# pd.plotting.scatter_matrix(stock_data[['open-close', 'high-low', 'target']], grid=True, diagonal='kde')

x_train, x_test, y_train, y_test = create_train_split_group(X, Y, split_ratio=0.8)

ols = linear_model.LinearRegression()
ols.fit(x_train, y_train)

print('Coefficients: \n', ols.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(y_train, ols.predict(x_train)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_train, ols.predict(x_train)))

# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, ols.predict(x_test)))

# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, ols.predict(x_test)))

# Use it to predict prices and calculate strategy returns
stock_data['predicted_signal'] = ols.predict(X)
stock_data['ZHGF_Returns'] = np.log(stock_data['close']/stock_data['close'].shift(-1))

print(stock_data.head())

def calculate_return(df, split_value, symbol):
    cum_stock_return = df[split_value:]['%s_Returns' % symbol].cumsum() * 100
    df['Strategy_Returns'] = df['%s_Returns' % symbol] * df['predicted_signal'].shift(1)
    return cum_stock_return

def calculate_strategy_return(df, split_value, symbol):
    cum_strategy_return = df[split_value:]['Strategy_Returns'].cumsum() * 100
    return cum_strategy_return

cum_stock_return = calculate_return(stock_data, split_value=len(x_train), symbol='ZHGF')
# print(cum_stock_return)
cum_strategy_return = calculate_strategy_return(stock_data, split_value=len(x_train), symbol='ZHGF')
# print(cum_strategy_return)

def plot_chart(cum_symbol_return, cum_strategy_return, symbol):
    plt.figure(figsize=(10, 5))
    plt.plot(cum_symbol_return, label='%s returns' % symbol)
    plt.plot(cum_strategy_return, label='Strategy returns')
    plt.legend()

    plt.savefig('zhgf_predict.png')

plot_chart(cum_stock_return, cum_strategy_return, 'ZHGF')


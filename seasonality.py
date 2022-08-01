from cProfile import label
from turtle import color
from jqdatasdk import *
from numpy import block

import pandas as pd
import matplotlib.pyplot as plt


# auth('13760263125', 'Turkey414')

# stock_price = get_price('603067.XSHG', start_date='2021-01-01', end_date='2022-07-29', frequency='daily', fields=None, skip_paused=False, fq='pre', count=None)

# stock_price.to_csv('zhgf.csv')

# print(stock_price.describe())

# stock_monthly_return = stock_price['close'].pct_change().groupby(
#     [stock_price['close'].index.year, stock_price['close'].index.month]
# ).mean()

# print(stock_monthly_return)

# for i in range(len)

stock_price = pd.read_csv('zhgf.csv', index_col=0)
# print(stock_price.describe())
# print(stock_price.index.dtype)
# print(stock_price.info())
# stock_monthly_return_pct = stock_price['close'].pct_change()
# print(stock_monthly_return_pct)


stock_price.index = pd.to_datetime(stock_price.index)
print(stock_price)

stock_monthly_return = stock_price['close'].pct_change().groupby(
    [stock_price['close'].index.year, stock_price['close'].index.month]
).mean()

print(stock_monthly_return)

# stock_monthly_return_list = stock_monthly_return.to_list
# print(len(stock_monthly_return))
# for i in range(len(stock_monthly_return)):
#     print(stock_monthly_return[i])

# stock_monthly_return_list = []

# for i in range(len(stock_monthly_return)):
#     stock_monthly_return_list.append(
#         {'month': stock_monthly_return.index[i][1],
#         'monthly_return': stock_monthly_return[stock_monthly_return.index[i]]}
#     )

# print(stock_monthly_return_list)
# stock_monthly_return_list=pd.DataFrame(stock_monthly_return_list,
#                                    columns=('month','monthly_return'))

# stock_monthly_return_list.boxplot(column='monthly_return', by='month')

# ax = plt.gca()
# labels = [item.get_text() for item in ax.get_xticklabels()]
# labels=['Jan','Feb','Mar','Apr','May','Jun',\
#       'Jul','Aug','Sep','Oct','Nov','Dec']
# ax.set_xticklabels(labels)
# ax.set_ylabel('ZHGF return')
# plt.tick_params(axis='both', which='major', labelsize=7)
# plt.title("ZHGF Monthly return 2021-2022")
# plt.suptitle("")
# plt.show()

def plot_rolling_statistics_ts(ts, titletext, ytest, filename, window_size=12):
    ts.plot(color='red', label='Original', lw=0.5)
    ts.rolling(window_size).mean().plot(
        color='blue', label='Rolling Mean'
    )
    ts.rolling(window_size).std().plot(
        color='black', label='Rolling Std'
    )

    plt.legend(loc='best')
    plt.ylabel(ytest)
    plt.title(titletext)
    plt.show(block=False)
    plt.savefig(filename)
    

# plot_rolling_statistics_ts(stock_monthly_return[1:],'ZHGF prices rolling mean and standard deviation','Monthly return', 'zhgf_month_return.png')


# print(stock_price['close'])
plot_rolling_statistics_ts(stock_price['close'],'ZHGF prices rolling mean and standard deviation','Daily price', 'zhgf_daily_close.png', 365)


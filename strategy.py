from jqdatasdk import *
import pandas as pd
import numpy as np

auth('13760263125', 'Turkey414')

stock_price = get_price('603067.XSHG', start_date='2021-01-01', end_date='2022-07-27', frequency='daily', fields=None, skip_paused=False, fq='pre', count=None)

print(stock_price)

lows = stock_price['low']
highs = stock_price['high']

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax1 = fig.add_subplot(111, ylabel='时代新材 price in rmb')
# highs.plot(ax=ax1, color='c', lw=2.)
# lows.plot(ax=ax1, color='y', lw=2.)

# # 阻力位与支撑位
# plt.hlines(highs.head(200).max(),lows.index.values[0],lows.index.values[-1],linewidth=2, color='g')
# plt.hlines(lows.head(200).min(),lows.index.values[0],lows.index.values[-1],linewidth=2, color='r')
# plt.axvline(linewidth=2,color='b',x=lows.index.values[200],linestyle=':')
# plt.show()


def trading_support_resistance(data, bin_width=20):
    data['sup_tolerance'] = pd.Series(np.zeros(len(data)))
    data['res_tolerance'] = pd.Series(np.zeros(len(data)))
    data['sup_count'] = pd.Series(np.zeros(len(data)))
    data['res_count'] = pd.Series(np.zeros(len(data)))
    data['sup'] = pd.Series(np.zeros(len(data)))
    data['res'] = pd.Series(np.zeros(len(data)))
    data['positions'] = pd.Series(np.zeros(len(data)))
    data['signal'] = pd.Series(np.zeros(len(data)))
    in_support=0
    in_resistance=0

    for x in range((bin_width - 1) + bin_width, len(data)):

        data_section = data[x - bin_width:x + 1]
        support_level = min(data_section['close'])
        resistance_level = max(data_section['close'])
        range_level = resistance_level - support_level
        data['res'][x] = resistance_level
        data['sup'][x] = support_level
        data['sup_tolerance'][x] = support_level + 0.2 * range_level
        data['res_tolerance'][x] = resistance_level - 0.2 * range_level

        if data['close'][x] >= data['res_tolerance'][x] and data['close'][x] <= data['res'][x]:
            in_resistance += 1
            data['res_count'][x] = in_resistance
        elif data['close'][x] <= data['sup_tolerance'][x] and \
                                    data['close'][x] >= data['sup'][x]:
            in_support += 1
            data['sup_count'][x] = in_support
        else:
            in_support = 0
            in_resistance = 0
        if in_resistance > 2:
            data['signal'][x] = 1
        elif in_support > 2:
            data['signal'][x] = 0
        else:
            data['signal'][x] = data['signal'][x-1]

    data['positions'] = data['signal'].diff()

trading_support_resistance(stock_price)

import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(111, ylabel='603067 price in $')
stock_price['sup'].plot(ax=ax1, color='g', lw=2.)
stock_price['res'].plot(ax=ax1, color='b', lw=2.)
stock_price['close'].plot(ax=ax1, color='r', lw=2.)
ax1.plot(stock_price.loc[stock_price.positions == 1.0].index,
       stock_price.close[stock_price.positions == 1.0],
       '^', markersize=7, color='k',label='buy')
ax1.plot(stock_price.loc[stock_price.positions == -1.0].index,
       stock_price.close[stock_price.positions == -1.0],
       'v', markersize=7, color='k',label='sell')
plt.legend()
plt.show()
# importing the dependencies modules
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from tqdm import tqdm
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")


def dark_mode():
	# The code will allow you to switch between the dark and light mode in graphs
	from matplotlib import cycler
	colors = cycler('color', ['#669FEE', '#66EE91', '#9988DD', '#EECC55', '#88BB44', '#FFBBBB'])
	plt.rc('figure', facecolor='#313233')
	plt.rc('axes', facecolor='#313233', edgecolor='None', axisbelow=True, grid=True, prop_cycle=colors,
	       labelcolor='gray')
	plt.rc('grid', color='#47a4a4')  # linestlye='solid')
	plt.rc('xtick', color='gray')
	plt.rc('ytick', direction='out', color='gray')
	plt.rc('legend', facecolor='#313233', edgecolor='#313233')
	plt.rc('text', color='#c9c9c9')


# Function to do RSI strategy
def rsi(val, neutral, window):
	"""

	Input:
	:param val: (type dataframe pandas): Entry values of the stock
	:param neutral:(float) Value of the neutrality, i.e No Action Zone
	:param window: (float) rolling period for RSI calculation

	Output: The function gives the returns of RSI strategy
	:return:
	"""
	# Print error is no Adj Close column is not present
	if "Adj Close" not in val.columns:
		ValueError("Error: No Adj Close column is present")

	val['rsi'] = ta.momentum.RSIIndicator(val['Adj Close'], window=window).rsi()

	"""Long Buy signal"""
	overbuy = 70
	neutral_buy = 50 + neutral

	# Put nan values for te signal long columns
	val['signal_long'] = np.nan
	val['yesterday_rsi'] = val['rsi'].shift(1)
	# We need to define open long signal (Rsi yes>55 and RSI today>55)
	val.loc[(val['rsi'] > neutral_buy) & (val['yesterday_rsi'] < neutral_buy), 'signal_long'] = 1
	# open long signal for false positive (Rsi yes>55 and RSI today<55)
	val.loc[(val['rsi'] < neutral_buy) & (val['yesterday_rsi'] > neutral_buy), 'signal_long'] = 0
	# open long signal for overbuy (Rsi yes>70 and RSI today<70)
	val.loc[(val['rsi'] < overbuy) & (val['yesterday_rsi'] > overbuy), 'signal_long'] = 0

	"""Short Sell signal"""
	oversell = 30
	neutral_sell = 50 - neutral

	# Put the values for the short signal columns
	val['signal_short'] = np.nan

	# We define the Open short signal (RSI yes>45 and RSI today<45)
	val.loc[(val['rsi'] < neutral_sell) & (val['yesterday_rsi'] > neutral_sell), 'signal_short'] = -1

	# We define the Close short signal(Rsi yes<45 and RSI today>45) False signal
	val.loc[(val['rsi'] > neutral_sell) & (val['yesterday_rsi'] < neutral_sell), 'signal_short'] = 0

	# We define the Close short signal (RSI yes<30 and RSI today>30) Oversell
	val.loc[(val['rsi'] < oversell) & (val['yesterday_rsi'] > oversell), 'signal_short'] = 0

	"""Compute the returns"""
	# Compute the percentage of the variation of the stock
	val['pct'] = val['Adj Close'].pct_change(1)

	# Calculate the position
	val['Position'] = (val['signal_short'].fillna(method='ffill') + val['signal_long'].fillna(method='ffill'))

	# Compute the returns
	val['return'] = val['pct'] * (val['Position'].shift(1))

	return val['return']


# import the data
data = yf.download('GOOG')

# test
returns = rsi(data['2010'], 5, 14)
returns.cumsum().plot(figsize=(15, 8))

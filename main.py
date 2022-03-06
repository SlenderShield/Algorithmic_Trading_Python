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
from hurst import compute_Hc
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
def rsi(value, neutral, window):
	"""

	Input:
	:param value: (type dataframe pandas): Entry values of the stock
	:param neutral:(float) Value of the neutrality, i.e No Action Zone
	:param window: (float) rolling period for RSI calculation

	Output: The function gives the returns of RSI strategy
	:return:
	"""
	# Print error is no Adj Close column is not present
	if "Adj Close" not in value.columns:
		ValueError("Error: No Adj Close column is present")

	value['rsi'] = ta.momentum.RSIIndicator(value['Adj Close'], window=window).rsi()

	"""Long Buy signal"""
	overbuy = 70
	neutral_buy = 50 + neutral

	# Put nan values for te signal long columns
	value['signal_long'] = np.nan
	value['yesterday_rsi'] = value['rsi'].shift(1)
	# We need to define open long signal (Rsi yes>55 and RSI today>55)
	value.loc[(value['rsi'] > neutral_buy) & (value['yesterday_rsi'] < neutral_buy), 'signal_long'] = 1
	# open long signal for false positive (Rsi yes>55 and RSI today<55)
	value.loc[(value['rsi'] < neutral_buy) & (value['yesterday_rsi'] > neutral_buy), 'signal_long'] = 0
	# open long signal for overbuy (Rsi yes>70 and RSI today<70)
	value.loc[(value['rsi'] < overbuy) & (value['yesterday_rsi'] > overbuy), 'signal_long'] = 0

	"""Short Sell signal"""
	oversell = 30
	neutral_sell = 50 - neutral

	# Put the values for the short signal columns
	value['signal_short'] = np.nan

	# We define the Open short signal (RSI yes>45 and RSI today<45)
	value.loc[(value['rsi'] < neutral_sell) & (value['yesterday_rsi'] > neutral_sell), 'signal_short'] = -1

	# We define the Close short signal(Rsi yes<45 and RSI today>45) False signal
	value.loc[(value['rsi'] > neutral_sell) & (value['yesterday_rsi'] < neutral_sell), 'signal_short'] = 0

	# We define the Close short signal (RSI yes<30 and RSI today>30) Oversell
	value.loc[(value['rsi'] < oversell) & (value['yesterday_rsi'] > oversell), 'signal_short'] = 0

	"""Compute the returns"""
	# Compute the percentage of the variation of the stock
	value['pct'] = value['Adj Close'].pct_change(1)

	# Calculate the position
	value['Position'] = (value['signal_short'].fillna(method='ffill') + value['signal_long'].fillna(method='ffill'))

	# Compute the returns
	value['return'] = value['pct'] * (value['Position'].shift(1))

	return value['return']


def beta_function(series):
	sp500 = yf.download('^GSPC')[['Adj Close']].pct_change(1)
	sp500.columns = ['SP500']

	# We concatenate the two dataframe
	value = pd.concat((series, sp500), axis=1)

	# We compute the beta
	beta = np.cov(value[[series.name, "SP500"]].dropna().values,rowvar=False)[0][1] / np.var(value["SP500"].dropna().values)
	return beta


def drawdown_function(series):
	cumulative_sum = series.dropna().cumsum() + 1  # multiplication coefficient
	# compute the max of the cumulative_sum
	# (1,3,1,1,7,1,2,) -> (1,3,3,3,7,7,7)
	running_max = np.maximum.accumulate(cumulative_sum)
	drawdowned = cumulative_sum / running_max - 1
	return drawdowned


def backTest(series):
	# Import the Bechmark
	sp500 = yf.download('^GSPC')['Adj Close'].pct_change()
	sp500.name = 'SP500'

	# We concatenate the two dataframe
	value = pd.concat([series, sp500], axis=1).dropna()

	# compute the drawdown and max drawdown
	drawdown = drawdown_function(series)
	max_drawdown = -np.min(drawdown)

	"""Plot the Graph"""
	fig, (cum, draw) = plt.subplots(1, 2, figsize=(20, 6))
	fig.suptitle("Backtesting", size=20)

	cum.plot(series.cumsum(), color='#39B3C7')
	cum.plot(value['SP500'].cumsum(), color='#B85A0F')

	cum.legend(["Portfolio", "Benchmark"])
	cum.set_title("Cumulative Returns", size=13)
	cum.set_ylabel("Cumulative Return in %", size=13)

	# Plot the drawdown
	draw.fill_between(drawdown.index, 0, drawdown * 100, color='#C73954', alpha=0.65)

	draw.set_title("Drawdown", size=15)
	draw.set_ylabel("Drawdown in %", size=13)

	# plot the graph
	plt.show()

	"""Compute the statistics"""
	# compute the sortino
	sortino = np.sqrt(252) * series.mean() / series.loc[series < 0].std()

	# compute the beta
	beta = np.cov(value[['return', 'SP500']].values, rowvar=False)[0][1] / np.var(value['SP500'].values)

	# compute the alpha
	alpha = 252 * (series.mean() - beta * series.mean())

	# Print the statistics

	print(f'Sortino: {np.round(sortino, 3)}')
	print(f'Beta: {np.round(beta, 3)}')
	print(f'Alpha: {np.round(alpha, 3)}')
	print(f'Sortino: {np.round(max_drawdown, 3)}')


# using Names CSV file find the best asset
def best_asset():
	asset = pd.read_csv('Names.csv')['Symbol']

	# Initialize the list
	Statistics = []
	col = []

	for fin in tqdm(asset):
		# check using yfinance
		try:
			print(fin)

			# Download the data of fin
			dataa = yf.download(fin).dropna()
			# Create a list of the statistics
			statistics = list()
			statistics.append(compute_Hc(dataa['Adj Close'])[0])  # Hurst Exponent
			statistics.append(np.sqrt(252) * dataa['Adj Close'].pct_change(1).dropna().std())  # volatility

			statistics.append(beta_function(dataa["Adj Close"].pct_change(1).dropna()))  # beta

			statistics.append(rsi(dataa, 5, 14).mean() * 252)  # RSI
			Statistics.append(statistics)
			col.append(fin)
		except Exception as e:
			pass

	# Create dataframe with all statistics
	dataframe = pd.DataFrame(Statistics, columns=['Hurst', 'Volatility', 'Beta', 'RSI'], index=col)
	cluster = pd.read_csv('Names.csv',index_col='Symbol')
	del cluster['Unnamed: 0']
	# Concat the type of asset and the statistics
	dataframe = pd.concat([cluster,dataframe],axis=1).dropna()
	# Plot the densities
	sns.displot(data=dataframe, x="RSI",kind='kde', hue="dummy")

	# limit the axis
	plt.xlim((-1.15,1.15))
	plt.show()
	print("Density of startegy returns by Hurst")
	# We are going to plot density of startegy returns by Hurst
	dataframe['Hurst_den'] = "Low"
	dataframe.loc[dataframe['Hurst']>0.55, 'Hurst_den'] = "High"
	# Plot the density
	sns.displot(data=dataframe, x="RSI", kind='kde', hue="Hurst_den")
	# limit and plot the graph
	plt.xlim(-1.15,1.15)
	plt.show()

	print("Density of startegy returns by Volatility")
	# We are going to plot density of startegy returns by Volatility
	dataframe['vol_den'] = "Low"
	dataframe.loc[dataframe['Volatility']>0.57, 'vol_den'] = "High"
	# Plot the density
	sns.displot(data=dataframe, x="RSI", kind='kde', hue="vol_den")
	# limit and plot the graph
	plt.xlim(-1.15,1.15)
	plt.show()

	print("Density of strategy returns by beta of Assest")
	# We are going to plot density of startegy returns by class of asset
	dataframe['beta_den'] = "Low"
	dataframe.loc[dataframe['Beta']>1, 'beta_den'] = "High"
	# Plot the density
	sns.displot(data=dataframe, x="RSI", kind='kde', hue="beta_den")
	# limit and plot the graph
	plt.xlim((-1.15,1.15))
	plt.show()


def optimization(data):
	# Statistical approach for Parameter
	# We list for the possible values of neutral and window
	neutral_values = [i*2 for i in range(10)]
	window_values = [i*2 for i in range(1,11)]

	# Set some dataset
	start_train,end_train = "2017-01-01",'2019-01-01'
	start_test,end_test = '2019-01-01','2020-01-01'
	start_valid,end_valid = '2020-01-01','2021-01-01'

	# Initialize the list
	result = []

	for i in range(len(neutral_values)):
		for j in range(len(window_values)):
			# Compute the return
			return_train = rsi(data.loc[start_train:end_train],neutral_values[i],window_values[j])
			return_test = rsi(data.loc[start_test:end_test],neutral_values[i],window_values[j])

			# Compute the sortino
			sortino_train = np.sqrt(252) * return_train.mean() / (return_train[return_train<0].std() + 0.00001)
			sortino_test = np.sqrt(252) * return_test.mean() / (return_test[return_test<0].std() + 0.00001)

			values = [neutral_values[i],window_values[j],sortino_train,sortino_test]
			result.append(values)

	dataframe = pd.DataFrame(result,columns=["Neutral","Window","Sortino_train","Sortino_test"])
	ordered_data = dataframe.sort_values(by=["Sortino_train"],ascending=False)
	for i in range(len(dataframe)):
		# Take the best
		best = ordered_data.iloc[0+i:1+i,:]

		# Extract the sortino
		sortino_train = best["Sortino_train"].values[0]
		sortino_test = best["Sortino_test"].values[0]

		# Take best neutral and window
		best_neutral = best["Neutral"].values[0]
		best_window = best["Window"].values[0]

		# If the best is found, we stop the loop
		if sortino_test > 0.5 and sortino_train > 0.5:
			break
		else:
			best_neutral = 0
			best_window = 0
			sortino_train = 0
			sortino_test = 0
	return [best_neutral,best_window,sortino_train,sortino_test]


# import the data
data = yf.download('AAPL')

# test
returns = rsi(data, 5, 14)
returns.cumsum().plot(figsize=(15, 8))
backTest(returns)

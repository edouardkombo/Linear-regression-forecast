# LinearRegression is a machine learning library for linear regression
from sklearn.linear_model import LinearRegression,LogisticRegression, Ridge

import pmdarima as pmd

# pandas and numpy are used for data manipulation
import pandas as pd
import numpy as np

# matplotlib and seaborn are used for plotting graphs
import matplotlib.pyplot as plt
# %matplotlib inline
plt.style.use('seaborn-darkgrid')

# yahoo finance is used to fetch data
import yfinance as yf
from datetime import datetime, timedelta, date
from textblob import TextBlob
import sys
import json

class LinearRegressionForecast:

    def __init__(self, symbol, start_date, end_date):
        self.currency_symbol=symbol + "=X"
        self.start_date=start_date
        self.end_date=end_date
        self.Df=[]
        self.training_set=0
        self.strategy=""
        self.returns=[]
        self.model = None
       
        self.roll = 1

        self.t=[]
        self.X=[]
        self.y=[]
        self.X_train=[]
        self.y_train=[]
        self.X_test=[]
        self.y_train=[]
        self.predicted_price=[]

    def get_data(self):
        df = yf.download(self.currency_symbol, self.start_date, self.end_date, auto_adjust=True)
        df['timestamp'] = df.index.values.astype(np.int64) // 10 ** 9
        return df


    def plot(self, column):
        self.Df[column].plot(figsize=(10, 7),color='r')
        plt.ylabel(self.currency_symbol + " Prices")
        plt.title(self.currency_symbol + " Price Series")
        plt.show()

    def build_model(self):
        self.Df['MA7'] = self.Df[self.strategy].rolling(window=self.roll).mean()
        self.Df['MA9'] = self.Df[self.strategy].rolling(window=self.roll).mean()
        self.Df['next_prediction'] = self.Df[self.strategy].shift(-1)
        self.Df = self.Df.dropna()

        self.X = self.Df[['MA7','MA9']]
        self.y = self.Df['next_prediction']

        self.t = self.training_set
        self.t = int(self.t*len(self.Df))

        self.X_train = self.X[:self.t]
        self.y_train = self.y[:self.t]

        self.X_test = self.X[self.t:]
        self.y_test = self.y[self.t:]

        # Create a linear regression model
        self.model = LinearRegression().fit(self.X_train, self.y_train)
        self.model = self.model
        print(self.currency_symbol + " Price (y) = %.2f * 3 Moving Average (x1) \
        + %.2f * 9 Moving Average (x2) \
        + %.2f (constant)" % (self.model.coef_[0], self.model.coef_[1], self.model.intercept_))


    def raw_prediction(self, plot=False):
        self.predicted_price = self.model.predict(self.X_test)
        self.predicted_price = pd.DataFrame(
        self.predicted_price, index=self.y_test.index, columns=['price'])

        if plot==True:
            self.predicted_price.plot(figsize=(10, 7))
            self.y_test.plot()
            plt.legend(['predicted_price', 'actual_price'])
            plt.ylabel(self.currency_symbol + " Price")
            plt.show()

    def stats(self, plot=False):
        # R square
        r2_score = self.model.score(self.X[self.t:], self.y[self.t:])*100
        float("{0:.2f}".format(r2_score))

        new_df = pd.DataFrame()

        new_df['current_strategy'] = self.Df[self.t:][self.strategy]
        new_df['next_strategy'] = self.predicted_price
        new_df['actual_vs_next'] = self.y_test
        new_df['symbol_returns'] = new_df['current_strategy'].pct_change().shift(-1)

        new_df['direction'] = np.where(new_df.next_strategy.shift(1) < new_df.next_strategy,1,0)

        new_df['strategy_returns'] = new_df.direction * new_df['symbol_returns']
        
        self.returns = new_df["strategy_returns"];

        if plot==True:
            ((new_df['strategy_returns']+1).cumprod()).plot(figsize=(10,7),color='g')
            plt.ylabel('Cumulative Returns')
            plt.show()

        'Sharpe Ratio %.2f' % (new_df['strategy_returns'].mean()/new_df['strategy_returns'].std()*(252**0.5))

    def prediction(self):
        data = self.Df
        data['MA7'] = data[self.strategy].rolling(window=self.roll).mean()
        data['MA9'] = data[self.strategy].rolling(window=self.roll).mean()

        del data['timestamp']
        del data['next_prediction']
        data = data.dropna()
        data['predicted_price'] = self.model.predict(data[['MA7','MA9']])
        data['direction'] = np.where(data.predicted_price.shift(1) < data.predicted_price,"Buy","Sell")
        return [data, self.returns]

    def run_prediction(self, strategy="Close", training_set=.75, plot=False):
        self.strategy=strategy
        self.training_set = training_set
        self.Df = self.get_data()
        self.build_model()
        self.raw_prediction(plot)
        self.stats(plot)
        return self.prediction()

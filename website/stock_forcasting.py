import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime
import matplotlib.pyplot as plt

class StockForecaster:
    def __init__(self, ticker):
        self.ticker = ticker
        self.df = None
        self.model = None
        self.scaler = None

    def plot_train_vs_test(self, y_train, y_train_pred, y_test, y_test_pred, save_path):
        """Plots the train vs test predictions and saves the graph as an image."""
        plt.figure(figsize=(12, 6))

        # Plotting training data
        plt.plot(self.df.index[len(self.df.index) - len(y_train):], y_train, label='True Train Data', color='blue')
        plt.plot(self.df.index[len(self.df.index) - len(y_train):], y_train_pred, label='Predicted Train Data',
                 color='orange')

        # Plotting testing data
        plt.plot(self.df.index[-len(y_test):], y_test, label='True Test Data', color='green')
        plt.plot(self.df.index[-len(y_test):], y_test_pred, label='Predicted Test Data', color='red')

        plt.title(f'{self.ticker} Train vs Test Predictions')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_historical_data(self, save_path):
        """Plots the historical closing prices and saves the graph as an image."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df['Close'], label=f'{self.ticker} Closing Price History')
        plt.title(f'{self.ticker} Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def plot_future_forecast(self, y_train, y_train_pred, y_test, y_test_pred, future_dates, forecasted_values, save_path):
        """Plots the future forecast and past data."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index[len(self.df.index) - len(y_train):], y_train, label='True Train Data')
        plt.plot(self.df.index[len(self.df.index) - len(y_train):], y_train_pred, label='Predicted Train Data')
        plt.plot(self.df.index[-len(y_test):], y_test, label='True Test Data')
        plt.plot(self.df.index[-len(y_test):], y_test_pred, label='Predicted Test Data')
        plt.plot(future_dates, forecasted_values, label='Future Predictions')
        plt.title(f'{self.ticker} Stock Price Prediction and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.savefig(save_path)
        plt.close()

    def fetch_data(self, start='2010-01-01', end=None):
        if end is None:
            end = datetime.today().strftime('%Y-%m-%d')

        self.df = yf.download(self.ticker, start=start, end=end)

        if self.df.empty:
            raise ValueError(f"No data fetched for ticker {self.ticker}.")
        return self.df

    def add_technical_indicators(self):
        self.df['SMA_50'] = SMAIndicator(close=self.df['Close'], window=50).sma_indicator()
        self.df['SMA_200'] = SMAIndicator(close=self.df['Close'], window=200).sma_indicator()
        self.df['EMA_20'] = EMAIndicator(close=self.df['Close'], window=20).ema_indicator()
        self.df['RSI'] = RSIIndicator(close=self.df['Close'], window=14).rsi()
        self.df.dropna(inplace=True)

    def prepare_features(self):
        self.df['day_of_week'] = self.df.index.dayofweek
        self.df['month'] = self.df.index.month
        self.df['year'] = self.df.index.year
        self.df['day_of_month'] = self.df.index.day
        self.df['quarter'] = self.df.index.quarter
        df1 = self.df[['Close', 'day_of_week', 'month', 'year', 'day_of_month', 'quarter', 'SMA_50', 'SMA_200', 'EMA_20', 'RSI']]
        return df1

    def train_model(self, df1):
        # Normalize the target variable (close price)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        df1['Close'] = self.scaler.fit_transform(df1[['Close']])

        X = df1.drop(['Close'], axis=1)
        y = df1['Close']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # XGBoost model
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=3, learning_rate=0.1)
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0]
        }
        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=3, verbose=1)
        grid_search.fit(X_train, y_train)

        # Best model
        self.model = grid_search.best_estimator_

        # Evaluate on test set
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Inverse transform predictions to return to original scale
        y_train_pred = self.scaler.inverse_transform(y_train_pred.reshape(-1, 1))
        y_test_pred = self.scaler.inverse_transform(y_test_pred.reshape(-1, 1))
        y_train = self.scaler.inverse_transform(y_train.values.reshape(-1, 1))
        y_test = self.scaler.inverse_transform(y_test.values.reshape(-1, 1))

        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

        return train_rmse, test_rmse, y_train, y_train_pred, y_test, y_test_pred

    def forecast_future(self, steps=30):
        # Start with the last available features and close price
        last_features = self.df.iloc[-1].copy()
        forecast = []
        future_dates = []

        for i in range(steps):
            # Predict the next close price
            prediction = self.model.predict(last_features[['day_of_week', 'month', 'year', 'day_of_month', 'quarter', 'SMA_50', 'SMA_200', 'EMA_20', 'RSI']].values.reshape(1, -1))
            predicted_close = prediction[0]

            # Update the forecast list and simulate the passage of one day
            forecast.append(predicted_close)
            future_dates.append(last_features.name + pd.DateOffset(1))
            last_features.name += pd.DateOffset(1)  # Simulate moving to the next date

            # Update technical indicators with the predicted close price
            last_features['Close'] = predicted_close
            last_features['SMA_50'] = SMAIndicator(close=pd.Series([predicted_close]), window=50).sma_indicator().iloc[-1]
            last_features['SMA_200'] = SMAIndicator(close=pd.Series([predicted_close]), window=200).sma_indicator().iloc[-1]
            last_features['EMA_20'] = EMAIndicator(close=pd.Series([predicted_close]), window=20).ema_indicator().iloc[-1]
            last_features['RSI'] = RSIIndicator(close=pd.Series([predicted_close]), window=14).rsi().iloc[-1]

        # Inverse transform the forecasted values to return to original scale
        forecasted_values = self.scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
        return future_dates, forecasted_values

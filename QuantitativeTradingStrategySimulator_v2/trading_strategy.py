import numpy as np
from arch import arch_model

class TradingStrategy:
    def __init__(self, lookback_period):
        self.lookback_period = lookback_period

    def generate_signals(self, data):
        raise NotImplementedError("Subclasses should implement this method")

    def execute_trades(self, data):
        data['Position'] = data['Signal'].shift()
        data['Returns'] = data['Close'].pct_change()
        data['StrategyReturns'] = data['Returns'] * data['Position']
        return data

    def evaluate_performance(self, data):
        total_return = data['StrategyReturns'].sum()
        sharpe_ratio = data['StrategyReturns'].mean() / data['StrategyReturns'].std() * np.sqrt(252)
        return total_return, sharpe_ratio


class MeanReversionStrategy(TradingStrategy):
    def generate_signals(self, data):
        data['Signal'] = 0
        data['RollingMean'] = data['Close'].rolling(window=self.lookback_period).mean()
        print(f"MeanReversionStrategy: data shape before signals: {data.shape}")
        data.iloc[self.lookback_period:, data.columns.get_loc('Signal')] = np.where(
            data['Close'].iloc[self.lookback_period:] > data['RollingMean'].iloc[self.lookback_period:], 1, -1)
        print(f"MeanReversionStrategy: data shape after signals: {data.shape}")
        return data


class MomentumStrategy(TradingStrategy):
    def generate_signals(self, data):
        data['Signal'] = 0
        data['RollingMean'] = data['Close'].rolling(window=self.lookback_period).mean()
        print(f"MomentumStrategy: data shape before signals: {data.shape}")
        print(f"MomentumStrategy: RollingMean shape: {data['RollingMean'].shape}")
        print(f"MomentumStrategy: Close shape: {data['Close'].shape}")

        close_prices = data['Close'].iloc[self.lookback_period:]
        rolling_mean = data['RollingMean'].iloc[self.lookback_period:]
        signal_array = data['Signal'].iloc[self.lookback_period:]

        print(f"MomentumStrategy: close_prices shape: {close_prices.shape}")
        print(f"MomentumStrategy: rolling_mean shape: {rolling_mean.shape}")
        print(f"MomentumStrategy: signal_array shape: {signal_array.shape}")

        data.iloc[self.lookback_period:, data.columns.get_loc('Signal')] = np.where(
            close_prices > rolling_mean, 1, 0)
        print(f"MomentumStrategy: data shape after first np.where: {data.shape}")

        data.iloc[self.lookback_period:, data.columns.get_loc('Signal')] = np.where(
            close_prices < rolling_mean, -1, data['Signal'].iloc[self.lookback_period:])
        print(f"MomentumStrategy: data shape after second np.where: {data.shape}")

        return data


class SMACrossoverStrategy(TradingStrategy):
    def __init__(self, short_window, long_window):
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, data):
        data['ShortMA'] = data['Close'].rolling(window=self.short_window).mean()
        data['LongMA'] = data['Close'].rolling(window=self.long_window).mean()
        data['Signal'] = 0
        print(f"SMACrossoverStrategy: data shape before signals: {data.shape}")
        print(f"SMACrossoverStrategy: ShortMA shape: {data['ShortMA'].shape}")
        print(f"SMACrossoverStrategy: LongMA shape: {data['LongMA'].shape}")

        short_ma = data['ShortMA'].iloc[self.short_window:]
        long_ma = data['LongMA'].iloc[self.short_window:]
        signal_array = data['Signal'].iloc[self.short_window:]

        print(f"SMACrossoverStrategy: short_ma shape: {short_ma.shape}")
        print(f"SMACrossoverStrategy: long_ma shape: {long_ma.shape}")
        print(f"SMACrossoverStrategy: signal_array shape: {signal_array.shape}")

        data.iloc[self.short_window:, data.columns.get_loc('Signal')] = np.where(
            short_ma > long_ma, 1, 0)
        print(f"SMACrossoverStrategy: data shape after first np.where: {data.shape}")

        data.iloc[self.short_window:, data.columns.get_loc('Signal')] = np.where(
            short_ma < long_ma, -1, data['Signal'].iloc[self.short_window:])
        print(f"SMACrossoverStrategy: data shape after second np.where: {data.shape}")

        return data


class PairsTradingStrategy(TradingStrategy):
    def __init__(self, lookback_period, spread_threshold):
        super().__init__(lookback_period)
        self.spread_threshold = spread_threshold

    def generate_signals(self, data1, data2):
        print(f"PairsTradingStrategy: Initial data1 shape: {data1.shape}, data2 shape: {data2.shape}")
        min_length = min(len(data1), len(data2))
        data1 = data1.iloc[-min_length:].copy()
        data2 = data2.iloc[-min_length:].copy()
        print(f"PairsTradingStrategy: Trimmed data1 shape: {data1.shape}, data2 shape: {data2.shape}")

        data1['Spread'] = data1['Close'] - data2['Close']
        data1['SpreadMean'] = data1['Spread'].rolling(window=self.lookback_period).mean()
        data1['SpreadStd'] = data1['Spread'].rolling(window=self.lookback_period).std()
        data1['ZScore'] = (data1['Spread'] - data1['SpreadMean']) / data1['SpreadStd']
        data1['Signal'] = 0
        print(f"PairsTradingStrategy: Before setting signals, data1 shape: {data1.shape}")
        data1.loc[data1['ZScore'] > self.spread_threshold, 'Signal'] = -1
        data1.loc[data1['ZScore'] < -self.spread_threshold, 'Signal'] = 1
        print(f"PairsTradingStrategy: After setting signals, data1 shape: {data1.shape}")
        return data1


class GARCHStrategy(TradingStrategy):
    def generate_signals(self, data):
        returns = data['Close'].pct_change().dropna()
        garch_model = arch_model(returns, vol='Garch', p=1, q=1)
        garch_fit = garch_model.fit(disp="off")
        data['Volatility'] = np.sqrt(garch_fit.conditional_volatility)
        data['Signal'] = 0
        print(f"GARCHStrategy: data shape before signals: {data.shape}")
        data.loc[data['Volatility'] > data['Volatility'].rolling(window=self.lookback_period).mean(), 'Signal'] = -1
        data.loc[data['Volatility'] < data['Volatility'].rolling(window=self.lookback_period).mean(), 'Signal'] = 1
        print(f"GARCHStrategy: data shape after signals: {data.shape}")
        return data

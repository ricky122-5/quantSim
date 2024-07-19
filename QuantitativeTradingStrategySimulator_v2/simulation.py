from data_handler import fetch_data, preprocess_data
from trading_strategy import MeanReversionStrategy, MomentumStrategy, SMACrossoverStrategy, PairsTradingStrategy, \
    GARCHStrategy


def run_simulation(ticker, strategies, start_date, end_date, pair_ticker=None):
    results = {}
    data = fetch_data(ticker, start_date, end_date)
    data = preprocess_data(data)

    if pair_ticker:
        data2 = fetch_data(pair_ticker, start_date, end_date)
        data2 = preprocess_data(data2)

    for strategy in strategies:
        print(f"Running {strategy.__class__.__name__}")
        if isinstance(strategy, PairsTradingStrategy):
            print(f"Data1 shape: {data.shape}, Data2 shape: {data2.shape}")
            strategy_data = strategy.generate_signals(data.copy(), data2.copy())
        else:
            print(f"Data shape: {data.shape}")
            strategy_data = strategy.generate_signals(data.copy())

        print(f"{strategy.__class__.__name__}: strategy_data shape before executing trades: {strategy_data.shape}")
        strategy_data = strategy.execute_trades(strategy_data)
        print(f"{strategy.__class__.__name__}: strategy_data shape after executing trades: {strategy_data.shape}")

        total_return, sharpe_ratio = strategy.evaluate_performance(strategy_data)
        results[strategy.__class__.__name__] = {
            'data': strategy_data,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio
        }

    return results

from simulation import run_simulation
from plotter import plot_results
from trading_strategy import MeanReversionStrategy, MomentumStrategy, SMACrossoverStrategy, PairsTradingStrategy, \
    GARCHStrategy

if __name__ == "__main__":
    ticker = 'AAPL'
    lookback_period = 20
    short_window = 10
    long_window = 50
    spread_threshold = 2
    start_date = '2020-01-01'
    end_date = '2023-01-01'

    strategies = [
        MeanReversionStrategy(lookback_period),
        MomentumStrategy(lookback_period),
        SMACrossoverStrategy(short_window, long_window),
        PairsTradingStrategy(lookback_period, spread_threshold),
        GARCHStrategy(lookback_period)
    ]

    results = run_simulation(ticker, strategies, start_date, end_date, pair_ticker='MSFT')
    plot_results(results)

    for strategy_name, result in results.items():
        print(
            f'{strategy_name}: Total Return = {result["total_return"]:.2f}, Sharpe Ratio = {result["sharpe_ratio"]:.2f}')

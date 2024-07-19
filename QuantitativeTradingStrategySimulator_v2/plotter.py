import matplotlib.pyplot as plt


def plot_results(results):
    plt.figure(figsize=(14, 7))

    for strategy_name, result in results.items():
        plt.plot(result['data']['Close'], label=f'{strategy_name} - Close Price')
        if 'RollingMean' in result['data']:
            plt.plot(result['data']['RollingMean'], label=f'{strategy_name} - Rolling Mean', linestyle='--')
        if 'Position' in result['data']:
            plt.plot(result['data'].index, result['data']['Position'] * result['data']['Close'],
                     label=f'{strategy_name} - Strategy Position', linestyle='--')

    plt.title('Trading Strategy Performance Comparison')
    plt.legend()
    plt.show()

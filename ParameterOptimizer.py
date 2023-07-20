import random
import math
import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt


population_size = 25
num_generations = 50
mutation_rate = 0.5

# Backtester data
capital = 1000
portfolio_return = None

# Analytics Dataframe to store all backtesting and price data
analytics_data:  pd.DataFrame




###########################
#     Define Strategy     #
###########################

# RSI STRATEGY #
#==============#
def rsi_strategy(length: int=12, lower: float=30, upper: float=70):
    global analytics_data
    try:
        # parameters like window size requires integers so select the nearest integer.
        length = int(round(length))

        # Calculate RSI using pandas-ta
        analytics_data['Rsi'] = ta.rsi(analytics_data['Close'], length=length)

        # Initialize 'signals' column with 'Hold'
        analytics_data['Signal'] = 0

        # Find the first occurrence of 'Buy' and 'Sell' signals
        for i in range(1, len(analytics_data)):
            if (analytics_data['Rsi'].iloc[i-1] > lower) and (analytics_data['Rsi'].iloc[i] <= lower):
                # buy signal
                analytics_data.at[i, 'Signal'] = 1
            elif (analytics_data['Rsi'].iloc[i-1] < upper) and (analytics_data['Rsi'].iloc[i] >= upper):
                # sell signal
                analytics_data.at[i, 'Signal'] = -1

        return strategy_runner()

    except:
        raise Exception

# parameters for supertrend strategy
'''
parameters = {'length': (1, 50)} 
'''


# SUPERTREND STRATEGY #
#=====================#
def supertrend_strategy(length: int=10, multiplier: float=2.5):
    global analytics_data
    try:
        # parameters like window size requires integers so select the nearest integer.
        length = int(round(length))

        # Calculate Supertrend indicator
        supertrend_data = ta.supertrend(high=analytics_data['High'],
                                        low=analytics_data['Low'],
                                        close=analytics_data['Close'],
                                        length=length, multiplier=multiplier)

        analytics_data['Supertrend'] = supertrend_data.iloc[:, 0]

        # Initialize 'signals' column with 'Hold'
        analytics_data['Signal'] = 0

        # Find the first occurrence of 'Buy' and 'Sell' signals
        for i in range(1, len(analytics_data)):
            if (analytics_data['Supertrend'].iloc[i-1] > analytics_data['Close'].iloc[i-1]) and (analytics_data['Supertrend'].iloc[i] <= analytics_data['Close'].iloc[i]):
                # buy signal
                analytics_data.at[i, 'Signal'] = 1
            elif (analytics_data['Supertrend'].iloc[i-1] < analytics_data['Close'].iloc[i-1]) and (analytics_data['Supertrend'].iloc[i] >= analytics_data['Close'].iloc[i]):
                # sell signal
                analytics_data.at[i, 'Signal'] = -1

        return strategy_runner()

    except:
        raise Exception

# parameters for supertrend strategy
'''
parameters = {'length': (1, 50),
              'multiplier': (1, 6)} 
'''

# CROSSOVER STRATEGY #
#====================#
def crossover_strategy(fast_length: int=50, slow_length: int=200):

    # parameters like window size requires integers so select the nearest integer.
    fast_length = int(round(fast_length))
    slow_length = int(round(slow_length))

    analytics_data['fast_ema'] = ta.ema(analytics_data['Close'], fast_length)
    analytics_data['slow_ema'] = ta.ema(analytics_data['Close'], slow_length)

    # Initialize 'signals' column with 'Hold'
    analytics_data['Signal'] = 0

    # Find the first occurrence of 'Buy' and 'Sell' signals
    for i in range(1, len(analytics_data)):
        if (analytics_data['fast_ema'].iloc[i-1] < analytics_data['slow_ema'].iloc[i-1]) and (analytics_data['fast_ema'].iloc[i] >= analytics_data['slow_ema'].iloc[i]):
            # buy signal
            analytics_data.at[i, 'Signal'] = 1
        elif (analytics_data['fast_ema'].iloc[i-1] > analytics_data['slow_ema'].iloc[i-1]) and (analytics_data['fast_ema'].iloc[i] <= analytics_data['slow_ema'].iloc[i]):
            # sell signal
            analytics_data.at[i, 'Signal'] = -1

    return strategy_runner()

# parameters for crossover strategy
'''
parameters = {'fast_length': (1, 100),
#              'slow_length': (50, 500)} 
'''



# SELECT STRATEGY TO FIND THE MAXIMIZATION PARAMETERS #
#=====================================================#
function_to_maximize = supertrend_strategy

# DEFINE PARAMETER DICTIONARY WITH MAXIMIZATION CONSTRAINTS TUPLE #
#=================================================================#
parameters = {'length': (1, 50),
              'multiplier': (1, 6)} 



########################################
#     GENETICS ALGORITHM FUNCTIONS     #
########################################

def generate_initial_population():
    """
    Generates the initial population of individuals.

    Returns:
        list: The initial population as a list of dictionaries, where each dictionary represents an individual with parameter-value pairs.
    """
    try:
        population = []
        for _ in range(population_size):
            individual = {
                param: random.uniform(min_val, max_val)
                for param, (min_val, max_val) in parameters.items()
            }
            population.append(individual)
        return population
    except:
        raise Exception

def evaluate_fitness(population):
    """
    Evaluates the fitness scores for the population.

    Args:
        population (list): The population of individuals to evaluate.

    Returns:
        list: The fitness scores for each individual in the population.
    """
    try:
        fitness_scores = []
        for individual in population:
            fitness_scores.append(function_to_maximize(**individual))
        return fitness_scores
    except:
        raise Exception

def tournament_selection(population: list, fitness_scores: list, num_parents: int) -> list:
    """
    Performs tournament selection to choose parents from the population.

    Args:
        population (list): The population of individuals.
        fitness_scores (list): The fitness scores corresponding to each individual in the population.
        num_parents (int): The number of parents to select.

    Returns:
        list: The selected parents as a list of individuals.
    """
    try:
        selected_parents = []
        for _ in range(num_parents):
            tournament = random.choices(list(range(population_size)), k=3)
            selected_parent = tournament[0]
            for competitor in tournament[1:]:
                if fitness_scores[competitor] > fitness_scores[selected_parent]:
                    selected_parent = competitor
            selected_parents.append(population[selected_parent])
        return selected_parents
    except:
        raise Exception

def crossover(parent1: dict, parent2: dict) -> dict:
    """
    Performs crossover between two parents to generate offspring.

    Args:
        parent1 (dict): The first parent individual as a dictionary of parameter-value pairs.
        parent2 (dict): The second parent individual as a dictionary of parameter-value pairs.

    Returns:
        dict: The offspring individual generated through crossover as a dictionary of parameter-value pairs.
    """
    try:
        offspring = {}
        for param in parameters:
            if random.random() < 0.5:
                offspring[param] = parent1[param]
            else:
                offspring[param] = parent2[param]
        return offspring
    except:
        raise Exception

def mutate(individual: dict) -> dict:
    """
    Performs mutation on an individual by randomly modifying its parameter values.

    Args:
        individual (dict): The individual to mutate as a dictionary of parameter-value pairs.

    Returns:
        dict: The mutated individual with updated parameter values.
    """
    try:
        for param in individual:
            if random.random() < mutation_rate:
                min_val, max_val = parameters[param]
                individual[param] = random.uniform(min_val, max_val)
        return individual
    except:
        raise Exception

def run_genetic_algorithm() -> dict:
    """
    Runs the genetic algorithm to optimize the parameters of a given function.

    Returns:
        dict: The optimized output, including the maximum fitness score ('delta_max') and the corresponding parameter values ('parameters').
    """
    try:
        # dictionary to save parameters best in each generation
        best_of_gens = {key: [] for key in parameters.keys()}
        best_parameters = {'delta':0,
                          'parameters':{}}

        # generate initial population
        population = generate_initial_population()

        for generation in range(num_generations):
            fitness_scores = evaluate_fitness(population)
            best_fitness = max(fitness_scores)
            best_individual = population[fitness_scores.index(best_fitness)]

            if (not math.isnan(best_fitness)) and (best_fitness>best_parameters['delta']):
                best_parameters['delta'] = best_fitness
                best_parameters['parameters'] = best_individual

            print(f"Generation {generation+1}: \n\t Best Fitness = {best_fitness:.2f}, Best Individual = {best_individual}")

            selected_parents = tournament_selection(population, fitness_scores, num_parents=2)
            offspring = []

            for i in range(0, population_size, 2):
                parent1 = selected_parents[i % len(selected_parents)]
                parent2 = selected_parents[(i + 1) % len(selected_parents)]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                offspring.extend([mutate(child1), mutate(child2)])
            population = offspring

        # Best parameters after each generation
        fitness_scores = evaluate_fitness(population)

        return best_parameters
    except:
        raise Exception
    



################################
#     BACKTESTER FUNCTIONS     #
################################

def backtest() -> float:
    """
    Backtests a trading strategy using the provided 'analytics_data' DataFrame and calculates the portfolio return.

    Returns:
        float: The portfolio return as a percentage.
    """

    global analytics_data, capital
    try:
        asset = 0
        profit = []
        market_returns = []
        inPosition = None
        m_asset = capital/analytics_data['Close'].iloc[0]
        invested_capital = 0

        for i in range(len(analytics_data)):
            market_returns.append(m_asset*analytics_data['Close'].iloc[i] - capital)

            if (inPosition is not None):
                # check if in a trade then wait for sell signal
                if inPosition:
                    # check for sell if in trade position
                    if (analytics_data['Signal'].iloc[i] == -1):
                        # sell order
                        inPosition = False
                        c_profit = (asset * analytics_data['Close'].iloc[i]) - capital
                        profit.append(c_profit)
                        invested_capital = invested_capital + capital

                    # no action is done if signal is buy or hold then also profit will varie with price
                    else:
                        # if already in trade and signals to hold then profit will vary with price
                        c_profit = (asset * analytics_data['Close'].iloc[i]) - capital
                        profit.append(c_profit)

                # while not in a trade wait for buy signal
                else:
                    # check for buy signal if not in trade position
                    if (analytics_data['Signal'].iloc[i] == 1):
                        # buy order
                        asset = capital / analytics_data['Close'].iloc[i]
                        inPosition = True
                        profit.append(0)

                    # the profit remains same for both sell and hold signal because we are not holding any asset
                    else:
                        # wait for a buy order, portfolio remains stagnant profit will remain same as previous
                        profit.append(profit[-1])

            else:
                if (analytics_data['Signal'].iloc[i] == 1):
                    # buy order
                    asset = capital / analytics_data['Close'].iloc[i]
                    inPosition = True

                profit.append(0) # profit remain 0 if signal is sell or hold because we are waiting for initial buying

        analytics_data['Strategy_return'] = profit
        analytics_data['Market_return'] = market_returns

        # Check if sum_of_invested_capital is non-zero before performing the division
        if invested_capital != 0:
            # we calculate the portfolio return by dividing the sum of profits by the sum of invested capital.
            # This adjustment is made because the same capital is used for each trade,
            # and we want to account for the total capital invested throughout the trading period.
            portfolio_return = (sum(profit) / invested_capital) * 100
        else:
            # Set portfolio_return to 0 when no capital is invested
            portfolio_return = 0

        return portfolio_return

    except:
        raise Exception

def backtest_plot() -> None:
    """
    Plots the market return and strategy return from the 'analytics_data' DataFrame.

    The function creates a figure with two subplots, where the first subplot represents the market return and
    the second subplot represents the strategy return.

    Raises:
        Exception: If an error occurs during the plotting process.
    """

    try:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

        # Plot data on the first subplot
        ax1.plot(analytics_data['Date'], analytics_data['Market_return'], 'r-', label='Market return')
        #ax1.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add horizontal line at y=0
        ax1.set_xlabel('date')
        ax1.set_ylabel('closes')
        ax1.legend()

        # Plot data on the second subplot
        ax2.plot(analytics_data['Date'], analytics_data['Strategy_return'], 'b-', label='Strategy return')
        #ax2.axhline(0, color='gray', linestyle='--', linewidth=1)  # Add horizontal line at y=0
        ax2.set_xlabel('date')
        ax2.set_ylabel('profit')
        ax2.legend()

        # Adjust spacing between subplots
        fig.tight_layout()

        # Display the plot
        plt.show()
    except:
        raise Exception

def calc_max_drawdown(data: pd.Series, is_close: bool=False):
    """
    Calculates the maximum drawdown from the provided data series.

    Args:
        data (pd.Series): The data series representing the values.
        is_close (bool, optional): Whether the data represents closing prices. Defaults to False.

    Returns:
        float: The maximum drawdown as a percentage.

    Raises:
        Exception: If an error occurs during the calculation.
    """

    try:
        if not is_close:
            data = data+capital
        max_data_stock = data.rolling(window=len(data), min_periods=1).max()
        dd_stock = data/max_data_stock - 1
        max_drawdown = dd_stock.rolling(window=len(data), min_periods=1).min()

        # return the maximum drawdown after multiplying with -ve to make it positive
        return -(max_drawdown.min()*100)
    except:
        raise Exception

def strategy_runner() -> float:
    """
    Runs the strategy by performing backtesting and calculates the RoMaD (Return over Maximum Drawdown).

    Returns:
        float: The RoMaD value.

    Raises:
        Exception: If an error occurs during the calculation.
    """

    global analytics_data

    portfolio_return = backtest()

    max_drawdown = calc_max_drawdown(analytics_data['Strategy_return'])
    #print(portfolio_return, max_drawdown)
    RoMaD = portfolio_return/max_drawdown

    # If RoMaD is NaN, returns -999999 which shows its a bad populaion
    if (math.isnan(RoMaD)):
        return -999999
    return RoMaD


run_genetic_algorithm()
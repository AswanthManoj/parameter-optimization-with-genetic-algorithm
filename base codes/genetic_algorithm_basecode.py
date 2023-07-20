import random
'''
# Genetic algorithm base code #
# =========================== #

# The function to be maximized is calculated based by 'f(x,y) = profit/drawdown'
# since profit is proportional to maximization aim and drawdown is inversly proportional to that maximization aim

# functions:
# ==========
#   mutate()
#   crossover()
#   calculate_output()
#   evaluate_fitness()
#   tournament_selection()
#   run_genetic_algorithm()
#   generate_initial_population()

# Define the parameters and their respective ranges
# param sets the bounds between min and max for the individial parameter
parameters = {
    'param1': (0, 10),
    'param2': (0, 5),
    'param3': (0, 1),
    'param4': (0, 0)
}

# Number of individuals in each generation
population_size = 100

# Number of generations
num_generations = 100

mutation_rate = 0.1

def return_over_maximum_drawdown(param1: int|float=0, param2: int|float=0, param3: int|float=0, param4: int|float=0) -> int|float:
    """
    return_over_maximum_drawdown
    ================
    Parameters:\n
    param1: int|float=0, param2: int|float=0, param3: int|float=0, param4: int|float=0\n
    \tFunction to calculate the 'return over maximum drawdown' (RoMaD) which is the average return in a given period for a portfolio, expressed as a proportion of the maximum drawdown level. 

    Returns:\n
    RoMaD: int|float
    \tReturn over maximum drawdown.
    """

    profit_return = 2
    maximum_drawdown = 3

    RoMaD = (profit_return/maximum_drawdown)*param1 + param2**3 - (3*param3)**2

    return RoMaD

def generate_initial_population() -> list:
    """
    generate_initial_population
    ===========================
    Parameters:\n
    None.\n
    \tFunction to initial population of random parameter pairs

    Returns:\n
    population: list\n
    \tGenerates list of initial population of random parameter pair.
    """

    population = []
    for _ in range(population_size):
        individual = {param: random.uniform(min_val, max_val) for param, (min_val, max_val) in parameters.items()}
        population.append(individual)
    return population

def evaluate_fitness(population: list) -> list:
    """
    evaluate_fitness
    ================
    Parameters:\n
    population: list
    \tFunction to evaluate the fitness score of each individual in the population based on the RoMaD

    Returns:\n
    fitness_scores: list\n
    \tFitness score list corresponding to each population.
    """

    fitness_scores = []
    for individual in population:
        fitness_scores.append(return_over_maximum_drawdown(**individual))
    return fitness_scores

def tournament_selection(population: list, fitness_scores: list, num_parents: int) -> list:
    """
    tournament_selection
    ====================
    Parameters:\n
    population: list, fitness_scores: list, num_parents: int\n
    \tFunction to select parents for the next generation using tournament selection

    Returns:\n
    selected_parents: list\n
    \tReturns the selected parents list
    """

    selected_parents = []
    for _ in range(num_parents):
        tournament = random.choices(list(range(population_size)), k=3)
        selected_parent = tournament[0]
        for competitor in tournament[1:]:
            if fitness_scores[competitor] > fitness_scores[selected_parent]:
                selected_parent = competitor
        selected_parents.append(population[selected_parent])
    return selected_parents

def crossover(parent1: dict, parent2: dict) -> dict:
    """
    crossover
    =========
    Parameters:\n
    parent1: dict, parent2: dict\n
    \tFunction to perform crossover between two parents to produce offspring

    Returns:\n
    offspring: dict\n
    \tReturns the offspring dict for the next generation
    """

    offspring = {}
    for param in parameters:
        if random.random() < 0.5:
            offspring[param] = parent1[param]
        else:
            offspring[param] = parent2[param]
    return offspring

def mutate(individual: dict, mutation_rate: float) -> dict:
    """
    mutate
    ======
    Parameters:\n
    individual: dict, mutation_rate: float\n
    \tFunction to perform mutation on an individual by randomly perturbing its parameter values

    Returns:\n
    individual: dict\n
    \tReturns the mutated individual
    """

    for param in individual:
        if random.random() < mutation_rate:
            min_val, max_val = parameters[param]
            individual[param] = random.uniform(min_val, max_val)
    return individual

def run_genetic_algorithm() -> dict:
    """
    run_genetic_algorithm
    =====================
    Parameters:\n
    None\n
    \tRuns the genetic algorithm to find the optimized parameters.

    Returns:\n
    optimized_output: dict\n
    \tReturns the best parameter settings.
    """
    optimized_output = {'delta_max':0,
                    'parameters':{'param1':0,
                                  'param2':0,
                                  'param3':0,
                                  'param4':0}}
    population = generate_initial_population()

    for generation in range(num_generations):
        fitness_scores = evaluate_fitness(population)
        best_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)]

        # Select the best individuals with fitness from all generations 
        if best_fitness > optimized_output['delta_max']:
            optimized_output['delta_max'] = best_fitness
            optimized_output['parameters'] = best_individual

        print(f"Generation {generation+1}: \n\t Best Fitness = {best_fitness:.2f}, Best Individual = {best_individual}")

        selected_parents = tournament_selection(population, fitness_scores, num_parents=2)
        offspring = []
        for i in range(0, population_size, 2):
            parent1 = selected_parents[i % len(selected_parents)]
            parent2 = selected_parents[(i + 1) % len(selected_parents)]
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            offspring.extend([mutate(child1, mutation_rate=mutation_rate), mutate(child2, mutation_rate=mutation_rate)])
        
        population = offspring

    # Best parameters after each generation
    fitness_scores = evaluate_fitness(population)

    return optimized_output
   

#optimized_output=run_genetic_algorithm()
#print("\nBest fitness settings for parameters\n",optimized_output)

'''

class GeneticAlgorithm():
    """
    A class implementing a Genetic Algorithm for optimization.

    Attributes:
        population_size (int): The size of the population in each generation.
        num_generations (int): The number of generations to run the algorithm.
        mutation_rate (float): The probability of mutation for each parameter.
        parameters (dict): A dictionary specifying the parameter names and their ranges.
        function (callable): The function to be optimized.

    Methods:
        generate_initial_population(): Generates the initial population of individuals.
        evaluate_fitness(population): Evaluates the fitness scores for the population.
        tournament_selection(population, fitness_scores, num_parents): Performs tournament selection to choose parents.
        crossover(parent1, parent2): Performs crossover between two parents to generate offspring.
        mutate(individual): Mutates an individual by randomly modifying its parameters.
        run_genetic_algorithm(): Runs the genetic algorithm and returns the optimized output.

    Example usage:
        def return_over_maximum_drawdown(param1, param2, param3, param4):
            RoMaD = (6 * param4 + param1) * param1 + param2**3 - (3 * param3)**2
            return RoMaD

        parameters = {
            'param1': (min_val1, max_val1),
            'param2': (min_val2, max_val2),
            'param3': (min_val3, max_val3),
            'param4': (min_val4, max_val4)
        }

        ga = GeneticAlgorithm(return_over_maximum_drawdown, parameters)
        output = ga.run_genetic_algorithm()

    """

    def __init__(self, function, parameters, population_size=100, num_generations=100, mutation_rate=0.1):
        """
        Initializes an instance of the GeneticAlgorithm class.

        Args:
            function (callable): The function to be optimized.
            parameters (dict): A dictionary specifying the parameter names and their ranges.
            population_size (int, optional): The size of the population in each generation. Default is 100.
            num_generations (int, optional): The number of generations to run the algorithm. Default is 100.
            mutation_rate (float, optional): The probability of mutation for each parameter. Default is 0.1.
        """

        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.parameters = parameters
        self.function = function

    def generate_initial_population(self):
        """
        Generates the initial population of individuals.

        Returns:
            list: The initial population as a list of dictionaries, where each dictionary represents an individual with parameter-value pairs.
        """
        try:
            population = []
            for _ in range(self.population_size):
                individual = {
                    param: random.uniform(min_val, max_val) 
                    for param, (min_val, max_val) in self.parameters.items()
                }
                population.append(individual)
            return population
        except:
            raise Exception

    def evaluate_fitness(self, population):
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
                fitness_scores.append(self.function(**individual))
            return fitness_scores
        except:
            raise Exception

    def tournament_selection(self, population: list, fitness_scores: list, num_parents: int) -> list:
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
                tournament = random.choices(list(range(self.population_size)), k=3)
                selected_parent = tournament[0]
                for competitor in tournament[1:]:
                    if fitness_scores[competitor] > fitness_scores[selected_parent]:
                        selected_parent = competitor
                selected_parents.append(population[selected_parent])
            return selected_parents
        except:
            raise Exception

    def crossover(self, parent1: dict, parent2: dict) -> dict:
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
            for param in self.parameters:
                if random.random() < 0.5:
                    offspring[param] = parent1[param]
                else:
                    offspring[param] = parent2[param]
            return offspring
        except:
            raise Exception

    def mutate(self, individual: dict) -> dict:
        """
        Performs mutation on an individual by randomly modifying its parameter values.

        Args:
            individual (dict): The individual to mutate as a dictionary of parameter-value pairs.

        Returns:
            dict: The mutated individual with updated parameter values.
        """
        try:
            for param in individual:
                if random.random() < self.mutation_rate:
                    min_val, max_val = self.parameters[param]
                    individual[param] = random.uniform(min_val, max_val)
            return individual
        except:
            raise Exception

    def run_genetic_algorithm(self) -> dict:
        """
        Runs the genetic algorithm to optimize the parameters of a given function.

        Returns:
            dict: The optimized output, including the maximum fitness score ('delta_max') and the corresponding parameter values ('parameters').
        """
        try:
            optimized_output = {'delta_max':0,
                            'parameters':{}}
            population = self.generate_initial_population()
            for generation in range(self.num_generations):
                fitness_scores = self.evaluate_fitness(population)
                best_fitness = max(fitness_scores)
                best_individual = population[fitness_scores.index(best_fitness)]

                # Select the best individuals with fitness from all generations
                if best_fitness > optimized_output['delta_max']:
                    optimized_output['delta_max'] = best_fitness
                    optimized_output['parameters'] = best_individual

                print(f"Generation {generation+1}: \n\t Best Fitness = {best_fitness:.2f}, Best Individual = {best_individual}")

                selected_parents = self.tournament_selection(population, fitness_scores, num_parents=2)
                offspring = []

                for i in range(0, self.population_size, 2):
                    parent1 = selected_parents[i % len(selected_parents)]
                    parent2 = selected_parents[(i + 1) % len(selected_parents)]
                    child1 = self.crossover(parent1, parent2)
                    child2 = self.crossover(parent2, parent1)
                    offspring.extend([self.mutate(child1), self.mutate(child2)])
                population = offspring

            # Best parameters after each generation
            fitness_scores = self.evaluate_fitness(population)

            return optimized_output
        except:
            raise Exception


def return_over_maximum_drawdown(param1, param2):
    RoMaD = param2/param1
    return RoMaD

parameters = {
    'param1': (1, 50),
    'param2': (1, 100),

}

ga = GeneticAlgorithm(return_over_maximum_drawdown, parameters)
output = ga.run_genetic_algorithm()
print(output)

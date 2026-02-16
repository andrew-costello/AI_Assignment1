import random
import statistics
import matplotlib.pyplot as plt
import time


N = 0 # Number of exams to be scheduled.
K = 0 # Number of available time slots.
M = 0 # Number of students.
E = [0][0] # E[i][j] = 1 if student j is taking exam i, otherwise 0

def read_instance(file_name):
    # Parse input file.
    with open(file_name, 'r') as file:
        # Read the first line to get N, K, M.
        N, K, M = map(int, file.readline().split())

        # Create a 2D matrix E with N rows and M columns, filled with 0's.
        E = [[0 for j in range(N)] for i in range(M)]

        for i in range(M):
            line = file.readline().split()

            for j in range(N):
                E[i][j] = int(line[j])
    
    return N, K, M, E

def initialize_population(pop_size, N, K):
    # Create random initial solutions.
    population = []

    for _ in range(pop_size):
        timetable = []

        for i in range(N):
            time_slot = random.randint(0, K-1) # Randomly assign a time slot to each exam.
            timetable.append(time_slot)

        population.append(timetable)

    return population

def evaluate_fitness(solution, E, N, M):
    # Calculate fitness value.
    hard_violations = 0
    soft_violations = 0

    for i in range(M):
        exam_slots = []

        for j in range(N):
            if E[i][j] == 1: # Student j is taking exam i.
                exam_slots.append(solution[j]) # Get the time slot for exam i.
        
        for a in range(len(exam_slots)):
            for b in range(a + 1, len(exam_slots)):
                if exam_slots[a] == exam_slots[b]: # Two exams in same slot.
                    hard_violations += 1
        
        exam_slots.sort() # Sort the time slots for soft constraint checking.
        for k in range(len(exam_slots) - 1):
            if exam_slots[k+1] == exam_slots[k] + 1: # Two exams in consecutive slots.
                soft_violations += 1

    fitness = (1000 * hard_violations) + soft_violations
    return fitness

def select_parents(population, E, N, M, tournament_size):
    # Choose parents for reproduction.
    def tournament():
        candidates = random.sample(population, tournament_size)
        best = candidates[0] # Assume first candidate is best.
        best_fitness = evaluate_fitness(best, E, N, M)

        for sol in candidates[1:]:
            fitness = evaluate_fitness(sol, E, N, M)

            if fitness < best_fitness: # Lower fitness is better.
                best = sol
                best_fitness = fitness

        return best

    parent1 = tournament()
    parent2 = tournament()
    return parent1, parent2

def crossover(parent1, parent2, crossover_rate):
    # Create offspring from two parents.
    N = len(parent1)

    if random.random() < crossover_rate:
        cut_point = random.randint(1, N-1) # Random cut point for crossover.
        child = parent1[:cut_point] + parent2[cut_point:] # Combine parts of both parents.
    else:
        child = parent1[:] # No crossover, child is a copy of parent1.

    return child

def mutate(solution, K, mutation_rate):
    # Apply random changes to a solution.
    if random.random() < mutation_rate:
        N = len(solution)

        exam = random.randint(0, N-1) # Randomly select an exam to mutate.
        new_time_slot = random.randint(0, K-1) # Randomly select a new time slot for the exam.
        solution[exam] = new_time_slot # Mutate the solution by changing the time slot for the selected exam.

    return solution

def run_ga(N, K, M, E, pop_size, generations, crossover_rate, mutation_rate, tournament_size):



    population = initialize_population(pop_size, N, K) # Initialize the population with random solutions.

    best_solution = None
    best_fitness = float('inf')
    fitness_history = [] # To track fitness over generations.
    
    for _ in range(generations):
        # Find the best solution in the current population.
        for solution in population:
            fitness = evaluate_fitness(solution, E, N, M)

            if fitness < best_fitness:
                best_solution = solution[:]
                best_fitness = fitness
            
        fitness_history.append(best_fitness) # Record the best fitness for this generation.
        new_population = [best_solution[:]] # Elitism: carry the best solution to the next generation.

        while len(new_population) < pop_size:
            parent1, parent2 = select_parents(population, E, N, M, tournament_size) # Select parents using tournament selection.
            child = crossover(parent1, parent2, crossover_rate) # Create a child through crossover.
            child = mutate(child, K, mutation_rate) # Mutate the child solution.
            new_population.append(child) # Add the child to the new population.
        
        population = new_population
    
    return best_solution, best_fitness, fitness_history

def count_hard_violations(solution, E, N, M):
    hard = 0

    for i in range(M):
        slots = []
        for j in range(N):
            if E[i][j] == 1:
                slots.append(solution[j])

        for a in range(len(slots)):
            for b in range(a + 1, len(slots)):
                if slots[a] == slots[b]:
                    hard += 1

    return hard


def count_soft_violations(solution, E, N, M):
    soft = 0

    for i in range(M):                      
        slots = []
        for j in range(N):                  
            if E[i][j] == 1:
                slots.append(solution[j])

        slots.sort()
        for k in range(len(slots) - 1):
            if slots[k + 1] == slots[k] + 1:
                soft += 1

    return soft

def main():
    # pop_size = 100  # Population size.
    # generations = 500  # Number of generations to run.
    # crossover_rate = 0.8  # Probability of crossover.
    # mutation_rate = 0.05  # Probability of mutation.
    # tournament_size = 3  # Number of candidates in tournament selection.
    #
    # filename = 'small-2.txt' # Input file containing the problem instance.
    # files= ['test_case1.txt', 'small-2.txt', 'medium-1.txt']
    # run_count = 1
    # for file in files:
    #     print("\n------Run", run_count , "with file:", file,"------")
    #     run_count= run_count + 1
    #     start_time = time.perf_counter()
    #     N, K, M, E = read_instance(filename) # Read the problem instance from the file.
    #     best_solution, best_fitness, fitness_history = run_ga(N, K, M, E,pop_size, generations, crossover_rate, mutation_rate, tournament_size) # Run the genetic algorithm to find the best solution.
    #     hard_violations = count_hard_violations(best_solution, E, N, M) # Count the number of hard constraint violations in the best solution.
    #     soft_violations = count_soft_violations(best_solution, E, N, M) # Count the number of soft constraint violations in the best solution.
    #     end_time = time.perf_counter()
    #     runtime = end_time - start_time
    #     print("Best Timetable:", best_solution) # Print the best timetable found.
    #     print("Best Fitness:", best_fitness) # Print the fitness of the best solution.
    #     print("Hard Violations:", hard_violations)
    #     print("Soft Violations:", soft_violations)
    #     print("Runtime:", runtime)
    #
    #     plt.plot(fitness_history)
    #     plt.xlabel("Generation")
    #     plt.ylabel("Best fitness")
    #     plt.title("GA Fitness Over Generations on " + file)
    #     plt.show()



#
#     pop_size_list = [50, 100, 200]
#     crossover_rate_list = [0.7, 0.8, 0.9]
#     mutation_rate=0.05
#     generations = 500  # Number of generations to run.
#     tournament_size = 3  # Number of candidates in tournament selection.
#
#     filename = 'small-2.txt'  # Input file containing the problem instance.
#     files= ['test_case1.txt', 'small-2.txt', 'medium-1.txt']
#
#     best_pop_size = 0
#     best_crossover_rate = 0
#     best_final_fitness = float('inf')
#
#     run_count=0
#
#     best_soft_violations = 0
#     best_hard_violations = 0
#     best_timetable = []
#
#     results=[]
#     repeats=1
#
#     for pop_size in pop_size_list:
#         for crossover_rate in crossover_rate_list:
#                 run_count += 1
#                 runtimes = []
#                 fitnesses = []
#
#                 start_time = time.perf_counter()
#
#                 N, K, M, E = read_instance(filename) # Read the problem instance from the file.
#                 best_solution, best_fitness, fitness_history = run_ga(N, K, M, E,pop_size, generations, crossover_rate, mutation_rate, tournament_size) # Run the genetic algorithm to find the best solution.
#                 hard_violations = count_hard_violations(best_solution, E, N, M) # Count the number of hard constraint violations in the best solution.
#                 soft_violations = count_soft_violations(best_solution, E, N, M) # Count the number of soft constraint violations in the best solution.
#                 end_time = time.perf_counter()
#                 runtime = end_time - start_time
#
#                 if best_fitness< best_final_fitness:
#                     best_final_fitness = best_fitness
#                     best_pop_size = pop_size
#                     best_crossover_rate = crossover_rate
#                     best_soft_violations = soft_violations
#                     best_hard_violations = hard_violations
#                     best_timetable= best_solution
#
#                 results.append((pop_size, crossover_rate,best_fitness,hard_violations,soft_violations,runtime))
#                 print("\n------Run", run_count , "with pop_size:", pop_size, "crossover_rate:", crossover_rate,"------")
#                 print("Best Timetable:", best_solution) # Print the best timetable found.
#                 print("Best Fitness:", best_fitness) # Print the fitness of the best solution.
#                 print("Hard Violations:", hard_violations)
#                 print("Soft Violations:", soft_violations)
#                 print("Runtime:", runtime)
#
#     print("\n-------All RESULTS-------")
#     for pop_size, crossover_rate, fitness, hard_violations, soft_violations, runtime in results:
#         print(f"Pop Size: {pop_size}, Crossover Rate: {crossover_rate}, Mutation Rate: {mutation_rate}, Fitness: {fitness}, Hard Violations: {hard_violations} Soft Violations: {soft_violations}, Runtime: {runtime}")
#
#
#     print("\n-------Best Combo-------")
#     print("Best Population Size:", best_pop_size)
#     print("Best Crossover Rate:", best_crossover_rate)
#     print("Best Mutation Rate:", mutation_rate)
#     print("Best Fitness:", best_final_fitness)
#     print("Best Hard Violations:", best_hard_violations)
#     print("Best Soft Violations:", best_soft_violations)
#     print("Best Timetable:", best_timetable)
#
#     N, K, M, E = read_instance(filename)
#     best_solution, best_fitness, fitness_history = run_ga(N, K, M, E,best_pop_size, generations, best_crossover_rate, mutation_rate, tournament_size) # Run the genetic algorithm to find the best solution.
#     plt.plot(fitness_history)
#     plt.xlabel("Generation")
#     plt.ylabel("Best fitness")
#     plt.title("GA Fitness Over Generations with Best Parameters")
#     plt.show()

    parameter_settings = [

        {  # Setting A
            "name": "Conservative",
            "pop_size": 70,
            "generations": 400,
            "crossover_rate": 0.7,
            "mutation_rate": 0.03,
            "tournament_size": 5
        },

        {  # Setting B
            "name": "Balanced",
            "pop_size": 100,
            "generations": 500,
            "crossover_rate": 0.8,
            "mutation_rate": 0.05,
            "tournament_size": 3
        },

        {  # Setting C
            "name": "Exploratory",
            "pop_size": 150,
            "generations": 600,
            "crossover_rate": 0.9,
            "mutation_rate": 0.1,
            "tournament_size": 2
        }
    ]

    filename="medium-1.txt"
    repeat=3
    run_count=0
    results=[]
    files=["test_case1.txt","small-2.txt", "medium-1.txt"]

    best_run_parameters=""
    best_runtime=0
    best_hard_violations=0
    best_soft_violations=0
    best_run_fitness = float('inf')

    best_mean_parameters=""
    best_mean_fitness = float("inf")
    best_std_fitness = float("inf")
    best_mean_runtime = float("inf")
    best_std_runtime = float("inf")

    worst_mean_parameters = ""
    worst_mean_fitness = float("-inf")
    worst_std_fitness = float("-inf")
    worst_mean_runtime = float("-inf")
    worst_std_runtime = float("-inf")



    print("\n------Running file:", filename,"------")
    for params in parameter_settings:
        parameter_setting_name=params["name"]
        print("\n---- Running Setting:",parameter_setting_name,"----")

        runtimes = []
        fitnesses = []

        for i in range(repeat):
            run = i + 1
            print("\n---- Run :", run, "----")




            pop_size = params["pop_size"]
            generations = params["generations"]
            crossover_rate = params["crossover_rate"]
            mutation_rate = params["mutation_rate"]
            tournament_size = params["tournament_size"]

            start_time = time.perf_counter()

            N, K, M, E = read_instance(filename) # Read the problem instance from the file.
            best_solution, best_fitness, fitness_history = run_ga(N, K, M, E,pop_size, generations, crossover_rate, mutation_rate, tournament_size) # Run the genetic algorithm to find the best solution.
            hard_violations = count_hard_violations(best_solution, E, N, M) # Count the number of hard constraint violations in the best solution.
            soft_violations = count_soft_violations(best_solution, E, N, M) # Count the number of soft constraint violations in the best solution.
            end_time = time.perf_counter()
            runtime = end_time - start_time

            if best_fitness < best_run_fitness:
                best_run_fitness = best_fitness
                best_run_parameters = params["name"]
                best_runtime=runtime
                best_hard_violations=hard_violations
                best_soft_violations=soft_violations

            fitnesses.append(best_fitness)
            runtimes.append(runtime)

            results.append(( parameter_setting_name,run,best_fitness,hard_violations,soft_violations,runtime))

            print("Best Timetable:", best_solution) # Print the best timetable found.
            print("Best Fitness:", best_fitness) # Print the fitness of the best solution.
            print("Hard Violations:", hard_violations)
            print("Soft Violations:", soft_violations)
            print("Runtime:", runtime)

        if len(fitnesses) > 1:
            fitness_std = statistics.stdev(fitnesses)
            runtime_std = statistics.stdev(runtimes)
        else:
            fitness_std = 0
            runtime_std = 0
        fitness_mean = statistics.mean(fitnesses)
        runtime_mean = statistics.mean(runtimes)

        if (fitness_mean < best_mean_fitness) or (
                fitness_mean == best_mean_fitness and fitness_std < best_std_fitness
        ):
            best_mean_parameters = parameter_setting_name
            best_mean_fitness = fitness_mean
            best_std_fitness = fitness_std
            best_mean_runtime = runtime_mean
            best_std_runtime = runtime_std

        if (fitness_mean > worst_mean_fitness) or (
                fitness_mean == worst_mean_fitness and fitness_std > worst_std_fitness
        ):
            worst_mean_parameters = parameter_setting_name
            worst_mean_fitness = fitness_mean
            worst_std_fitness = fitness_std
            worst_mean_runtime = runtime_mean
            worst_std_runtime = runtime_std

        print("\nStats for", parameter_setting_name)
        print("Average Fitness:", fitness_mean)
        print("Fitness Std Dev:", fitness_std)
        print("Average Runtime:", runtime_mean)
        print("Runtime Std Dev:", runtime_std)


    print("\n-------All RESULTS-------")

    for  parameter_setting_name, run, fitness, hard_violations, soft_violations, runtime in results:
        print(f"Params Setting: {parameter_setting_name}, Run: {run } Fitness: {fitness}, Hard Violations: {hard_violations} Soft Violations: {soft_violations}, Runtime: {runtime}")

    print ("\n-------Best Run-------")
    print("Best Run Setting:", best_run_parameters)
    print("Best Run Fitness:", best_run_fitness)
    print("Best Run Runtime:", best_runtime)
    print("Best Run Hard Violations:", best_hard_violations)
    print("Best Run Soft Violations:", best_soft_violations)


    print("\n-----BEST PARAMETER SETTING-----")
    print("Best Setting:", best_mean_parameters)
    print("Mean Fitness:", best_mean_fitness)
    print("Fitness Std Dev:", best_std_fitness)
    print("Mean Runtime:", best_mean_runtime)
    print("Runtime Std Dev:", best_std_runtime)

    print("\n-----WORST PARAMETER SETTING-----")
    print("Worst Setting:", worst_mean_parameters)
    print("Mean Fitness:", worst_mean_fitness)
    print("Fitness Std Dev:", worst_std_fitness)
    print("Mean Runtime:", worst_mean_runtime)
    print("Runtime Std Dev:", worst_std_runtime)

    best_params = next(param for param in parameter_settings if param["name"] == best_mean_parameters)

    best_pop_size = best_params["pop_size"]
    best_generations = best_params["generations"]
    best_crossover_rate = best_params["crossover_rate"]
    best_mutation_rate = best_params["mutation_rate"]
    best_tournament_size = best_params["tournament_size"]

    N, K, M, E = read_instance(filename)
    best_solution, best_fitness, fitness_history = run_ga(N, K, M, E,best_pop_size, best_generations, best_crossover_rate, best_mutation_rate, best_tournament_size) # Run the genetic algorithm to find the best solution.
    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(f"GA Fitness Over Generations (Best Setting: {best_mean_parameters})")
    plt.show()

if __name__ == "__main__":
    main()
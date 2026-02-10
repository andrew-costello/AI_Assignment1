import random
import matplotlib.pyplot as plt

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
def run_ga(N, K, M, E):
    # Main genetic algorithm loop.
    pop_size = 100 # Population size.
    generations = 500 # Number of generations to run.
    crossover_rate = 0.8 # Probability of crossover.
    mutation_rate = 0.05 # Probability of mutation.
    tournament_size = 5 # Number of candidates in tournament selection.

    population = initialize_population(pop_size, N, K) # Initialize the population with random solutions.

    best_solution = None
    best_fitness = float('inf')
    fitness_history = [] # To track fitness over generations.

    # Tournament selection for one parent
    def pick_parent():
        candidates = random.sample(population, tournament_size)
        best = candidates[0]
        best_fitness = evaluate_fitness(best, E, N, M)

        for sol in candidates[1:]:
            fitness = evaluate_fitness(sol, E, N, M)

            if fitness < best_fitness:
                best = sol
                best_fitness = fitness

        return best
    
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
            parent1 = pick_parent() # Select the first parent using tournament selection.
            parent2 = pick_parent() # Select the second parent using tournament selection.
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
    filename = 'small-2.txt' # Input file containing the problem instance.

    N, K, M, E = read_instance(filename) # Read the problem instance from the file.
    best_solution, best_fitness, fitness_history = run_ga(N, K, M, E) # Run the genetic algorithm to find the best solution.
    hard_violations = count_hard_violations(best_solution, E, N, M) # Count the number of hard constraint violations in the best solution.
    soft_violations = count_soft_violations(best_solution, E, N, M) # Count the number of soft constraint violations in the best solution.

    print("Best Timetable:", best_solution) # Print the best timetable found.
    print("Best Fitness:", best_fitness) # Print the fitness of the best solution.
    print("Hard Violations:", hard_violations)
    print("Soft Violations:", soft_violations)

    plt.plot(fitness_history)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title("GA Fitness Over Generations")
    plt.show()

if __name__ == "__main__":
    main()
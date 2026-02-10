import random

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
    #Choose parents for reproduction.
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

def crossover(parent1, parent2):
#Create offspring from two parents.


def mutate(solution):
#Apply random changes to a solution.


def run_ga():
#Main genetic algorithm loop.


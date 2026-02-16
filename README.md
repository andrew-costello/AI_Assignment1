## Problem Definitions
N is the number of exams
K is the number of slots
M is the number of students
E is the enrollment Matrix

---

## Key Functions

### def read_instance(file_name)
This function reads in the test case file and returns the values for N, K, M, E.

### def initialize_population(pop_size, N, K)
This function initialises the population of random timetables.

### def evaluate_fitness(solution, E, N, M)
Takes in one timetable solution alongside E, N, and M and counts up the amount of soft and hard violations to produce a fitness score.

### def select_parents(population, E, N, M, tournament_size)
This function uses an inner tournament function to evaluate the fitness of candidate solutions in the selected tournament size.
This function returns two parent candidates.

### def crossover(parent1, parent2, crossover_rate)
Takes in two parents and the crossover rate.
Uses a random single point cross over to find the crossover point.
A child is only a combination of both its parents if the crossover rate is met if not it is a copy of parent1.

### def mutate(solution, K, mutation_rate)
Selects a random timeslot in a timetable and randomly changes/mutates it.

### def run_ga(N, K, M, E, pop_size, generations, crossover_rate, mutation_rate, tournament_size):
Takes in N, K, M, E and all the parameters.
Runs the genetic algorithim using all the funtions previously mentioned.
- initialize population
- track thebest fitness and the best solution (elitism)
- repeat for specified generations
  - evaluate fitness of all solutions
  - Update best solution if needed
  - Record best fitness
  - Create a new population starting with the best solution (elitism)
  - Select parents and generate children accordingly
  - Apply mutation
  - replace old population with the new one


### def main()
Sets up three different parameter settings Conservative, Balanced and Exploratory. Changing the filename variabel is how someone chooses which testcase to run. The code runs the slected test case with each parameter setting three times in order to evaluate consistency. The testing captures the mean fitness and standard deviation for each run. The stats for best singular run alongside the overall best and worst parameter settings are all printed. A plot of generations over best fitness is then made



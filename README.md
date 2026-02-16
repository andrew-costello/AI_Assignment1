N is the number of exams
K is thee number of slots
M is the number of students
E is the enrollment Matrix

Key Functions

def read_instance(file_name)
This function reads in the test case file and returns the values for N, K, M E

def initialize_population(pop_size, N, K)
This function inialises the population of random timetables

def evaluate_fitness(solution, E, N, M)
Takes in one timetable solution alongside E, N, and K and counts up the amount of soft and hard violations to produce a fitness score

def select_parents(population, E, N, M, tournament_size)
This function uses an inner tournament function to evaluate the fitness of candidate solutions in the selected tournament size
This function returns two parent candidates

def crossover(parent1, parent2, crossover_rate)
Takes in tow parents and the crossover rate.
Uses a random single point cross over to find the crossover point
A child is only a combination if both its parents if the crossover rate is met if not it is a copy of parent1

def mutate(solution, K, mutation_rate)
Selects a random timeslot in it a timetable and randomly changes/mutates it

def run_ga(N, K, M, E, pop_size, generations, crossover_rate, mutation_rate, tournament_size):
Takes in N, K, M, E and all the parameters
Runs the genetic algorithim using all the funtions previoulsy mentioned

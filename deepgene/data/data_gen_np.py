import os
import sys
from math import (exp, log)

import msprime
import demes

from sklearn.utils import shuffle
import numpy as np
from scipy import stats

from numpy import random as rnd
from math import log, exp

N = 32  # int(sys.argv[4])

T_max =  20 # 20 # 2500 # 20  # 400_000
coeff = np.log(T_max) / (N - 1)
MAGIC_COEFF = 550 # 2_000 ## 1_000 for 169302 as T_max, 2_000 for 338605
cut_coeff = 400.0

limits = [np.exp(coeff * i) for i in range(N)]
limits = [MAGIC_COEFF * (np.exp(i * np.log(1 + 10 * T_max) / N) - 1)
          for i in range(N)]

# limits = [cut_coeff] + [l for l in limits if l > cut_coeff]
limits = [
0.0,
2900,
6000,
9500,
15000,
18000,
30000,
42000
]

limits = [
    0,
    700,
    1500,
    2200,
    3000,
    3400,
    3800,
    4200,   
    4700,
    5200,
    5700,
    6200,
    7200,
    8500,
    9800,
    11000,
    12500,
    14000,  
    15500,
    16500,
    17500,
    18500,
    21000,
    24000,
    27000,
    31000,
    40000,
    55000,
    75000,
    95000,
    120000,
    10000000,
    20000000
]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

L_HUMAN = 30_000_000  # int(float(sys.argv[3])) #30_000_000
# RHO_HUMAN = 1.38 * 1e-8 #-9
# MU_HUMAN = 1.28 * 1e-8 #-8

RHO_HUMAN = 1.6 * 10e-9
MU_HUMAN = 1.25 * 10e-8

RHO_LIMIT = (1.6 * 10e-10, 1.6 * 10e-8)
MU_LIMIT = (1.25 * 10e-7, 1.25 * 10e-9)

NUMBER_OF_EVENTS_LIMITS = (1, 20)
LAMBDA_EXP = 20_000

POPULATION = 10_000
POPULATION_COEFF_LIMITS = (0.5, 1.5)

MIN_POPULATION_NUM = 1_000
MAX_POPULATION_NUM = 120_000

POPULATION_SIZE_LIMITS = (MIN_POPULATION_NUM, MAX_POPULATION_NUM)

lambda_exp = 500
coal_limits = .0001  # 0.999

POPULATION_COEFF_LIMIT_COMPLEX = [1.0, 2.0]


def give_rho() -> float:
    return np.random.uniform(*RHO_LIMIT)


def give_mu() -> float:
    return np.random.uniform(*MU_LIMIT)


def give_random_coeff(mean=.128, var=.05) -> float:
    return np.random.beta(.1, .028) * .0128


def give_random_rho(base=RHO_HUMAN) -> float:
    return np.random.uniform(0.0001, 100, 1)[0] * base


def give_population_size() -> int:
    return int(np.random.uniform(*[1_000, 29_000]))


def generate_demographic_events_complex(population: int = None, random_seed: int = 42) -> 'msprime.Demography':
    if not population:
        population = give_population_size()

    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=population)

    last_population_size = population
    T = 0
    coal_probability = 0.0
    coal_probability_list = []
    non_coal_probability = 1.0

    while T < 420_000:
        t = np.random.exponential(lambda_exp)
        T += t

        # last_population_size = max(last_population_size * np.random.uniform(*POPULATION_COEFF_LIMITS),
        #                           MIN_POPULATION_NUM)

        coeff = (np.random.uniform(*POPULATION_COEFF_LIMIT_COMPLEX)
                 ) ** (np.random.choice([-1, 1]))
        # print(last_population_size)
        last_population_size = min(
            max(last_population_size * coeff, MIN_POPULATION_NUM), MAX_POPULATION_NUM)

        demography.add_population_parameters_change(
            T, initial_size=last_population_size)

        coal_probability = non_coal_probability + t / last_population_size
        coal_probability_list.append(coal_probability)
        non_coal_probability = non_coal_probability + \
                               (-t / last_population_size)
    return demography


def generate_demographic_events(population: int = None, random_seed: int = 42) -> 'msprime.Demography':
    if not population:
        population = give_population_size()
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=population)

    number_of_events = np.random.randint(*NUMBER_OF_EVENTS_LIMITS)

    times = sorted(np.random.exponential(LAMBDA_EXP, size=number_of_events))

    last_population_size = population
    for t in times:
        last_population_size = max(last_population_size * np.random.uniform(*POPULATION_COEFF_LIMITS),
                                   MIN_POPULATION_NUM)
        demography.add_population_parameters_change(
            t, initial_size=last_population_size)

    return demography


def get_const_demographcs(population: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=population)

    return demography
def get_test_demographcs_mew(population: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    demography = msprime.Demography()

    demography.add_population(name="A", initial_size=population)
    demography.add_population_parameters_change(40_000, initial_size=0.1 * population)
    demography.add_population_parameters_change(240_000, initial_size=1. * population)
    demography.add_population_parameters_change(800_000, initial_size=.5 * population)
    demography.add_population_parameters_change(4_000_000, initial_size=1. * population)
    demography.add_population_parameters_change(80_000_000, initial_size=2. * population)

    return demography

def get_test_demographcs(population: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    demography = msprime.Demography()

    demography.add_population(name="A", initial_size=population)
    demography.add_population_parameters_change(400, initial_size=0.1 * population)
    demography.add_population_parameters_change(2400, initial_size=1. * population)
    demography.add_population_parameters_change(8000, initial_size=.5 * population)
    demography.add_population_parameters_change(40_000, initial_size=1. * population)
    demography.add_population_parameters_change(80_000, initial_size=2. * population)

    return demography


def get_test_demographcs_2(population: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    demography = msprime.Demography()

    demography.add_population(name="A", initial_size=population)
    for i in range(32):
        demography.add_population_parameters_change(i*1_000, initial_size=((.5 + 0.5*i%2)* population))

    demography.add_population_parameters_change(40_000, initial_size=1. * population)
    demography.add_population_parameters_change(80_000, initial_size=2. * population)

    return demography

def get_humanlike(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    
    # ms = "-eN 0.1 0.1 -eN 0.6 1 -eN  2 0.5 -eN 10 1 -eN 20 2"
    # graph = demes.from_ms(ms, N0=population, deme_names=["A"])
    # emography = msprime.Demography.from_demes(graph)

    demography = msprime.Demography()

    coeff = 25 

    demography.add_population(name="A", initial_size=initial_size)
    demography.add_population_parameters_change(4000/25, initial_size=0.1 * initial_size)
    demography.add_population_parameters_change(24000/25, initial_size=1. * initial_size)
    demography.add_population_parameters_change(80000/25, initial_size=.5 * initial_size)
    demography.add_population_parameters_change(400000/25, initial_size=1. * initial_size)
    demography.add_population_parameters_change(800000/25, initial_size=2. * initial_size)
    return demography

def get_humanlike(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    demography = msprime.Demography()

    demography.add_population(name="A", initial_size=initial_size)
    demography.add_population_parameters_change(2900, initial_size=0.5 * initial_size)
    demography.add_population_parameters_change(6000, initial_size=1. * initial_size)
    demography.add_population_parameters_change(15000, initial_size=.5 * initial_size)
    demography.add_population_parameters_change(18000, initial_size=1. * initial_size)
    demography.add_population_parameters_change(30000, initial_size=2. * initial_size)

    return demography


def get_humanlike_2(initial_size: int = 10_000, random_seed: int = 42) -> msprime.Demography:
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)

    N0 = 10_000  # фиксируем N0
    demography.add_population_parameters_change(time=1000,  population="A", initial_size=0.2 * N0)
    demography.add_population_parameters_change(time=3500,  population="A", initial_size=3.0 * N0)
    demography.add_population_parameters_change(time=10000, population="A", initial_size=1.5 * N0)

    return demography

def get_humanlike_2_1(initial_size: int = 10_000, random_seed: int = 42) -> msprime.Demography:
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)

    N0 = 10_000  # фиксируем N0
    demography.add_population_parameters_change(time=1800,  population="A", initial_size=0.2 * N0)
    demography.add_population_parameters_change(time=3500,  population="A", initial_size=3.0 * N0)
    demography.add_population_parameters_change(time=10000, population="A", initial_size=1.5 * N0)

    return demography


def get_random_events_by_points(population: int = 10_000, random_seed: int = 42, points = limits ) -> 'msprime.Demography':
    # пределы для Ne
    a, b = 0.1, 2.0
    
    demography = msprime.Demography()
    for (i, point) in enumerate(points):
        coeff = np.random.uniform(a, b)
        if i == 0: 
            demography.add_population(name="A", initial_size = coeff * population)
        else:
            demography.add_population_parameters_change(point, initial_size=coeff * population)
    return demography



def get_exp_growth(population: int = 10_000, grow_rate = 1.0 , random_seed: int = 42, points = limits) -> 'msprime.Demography':
    # пределы для Ne
    a, b = 0.1, 2.0
    
    cur_population = population
    demography = msprime.Demography()
    for (i, point) in enumerate(points):
        coeff = np.random.uniform(a, b)
        if i == 0: 
            demography.add_population(name="A", initial_size = cur_population)
        else:
            cur_population = cur_population * grow_rate
            demography.add_population_parameters_change(point, initial_size=cur_population)
    return demography


def get_dem_ZZZ(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    
    demography = msprime.Demography()

    coeff = 2.0
    
    demography.add_population(name="A", initial_size=initial_size)
    for l in limits:
        demography.add_population_parameters_change(l, initial_size=initial_size*coeff)
        if coeff == 2.0: 
            coeff = 1.0
        else:
            coeff = 2.0

    return demography


def get_zigzag(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    
    demography = msprime.Demography()
    
    demography.add_population(name="A", initial_size=initial_size)
    
    demography.add_population_parameters_change(1500, initial_size=initial_size*0.8)
    demography.add_population_parameters_change(2200, initial_size=initial_size*0.6)
    demography.add_population_parameters_change(3000, initial_size=initial_size*0.4)

    demography.add_population_parameters_change(3400, initial_size=initial_size*0.6)
    demography.add_population_parameters_change(3800, initial_size=initial_size*0.8)
    demography.add_population_parameters_change(4200, initial_size=initial_size*1.0)


    demography.add_population_parameters_change(9800, initial_size=initial_size*1.25)
    demography.add_population_parameters_change(11000, initial_size=initial_size*1.5)
    demography.add_population_parameters_change(12500, initial_size=initial_size*2.0)

    demography.add_population_parameters_change(14000, initial_size=initial_size*1.5)
    demography.add_population_parameters_change(15500, initial_size=initial_size*1.25)
    demography.add_population_parameters_change(16500, initial_size=initial_size*1.0)    

    demography.add_population_parameters_change(21000, initial_size=initial_size*0.8)
    demography.add_population_parameters_change(24000, initial_size=initial_size*0.6)
    demography.add_population_parameters_change(27000, initial_size=initial_size*0.4)
    
    return demography


current_K = 0
current_K_plus = 0

def get_one_of_two(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
  
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)

    global current_K
    if current_K%2==0:
        demography.add_population_parameters_change(2900, initial_size=0.5 * initial_size)
        demography.add_population_parameters_change(6000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(15000, initial_size=.5 * initial_size)
        demography.add_population_parameters_change(18000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(30000, initial_size=2. * initial_size)
    

    current_K += 1
    return demography


def get_one_of_many(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':

    global current_K
    points = limits
    population = 10_000
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)   

    #current_K = np.random.choice([0,1,2,3,4])

    if current_K == 0:
        current_K = 1
    elif current_K == 1:
        demography.add_population_parameters_change(2900, initial_size=0.5 * initial_size)
        demography.add_population_parameters_change(6000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(15000, initial_size=.5 * initial_size)
        demography.add_population_parameters_change(18000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(30000, initial_size=2. * initial_size)

        current_K = 2
    elif current_K == 2:
        a, b = 0.1, 2.0    
        for (i, point) in enumerate(points):
            coeff = np.random.uniform(a, b)
            if i != 0: 
                demography.add_population_parameters_change(point, initial_size=coeff * population)
        current_K = 3
    elif current_K == 3:
        grow_rate = 1.1
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K = 4
    elif current_K == 4:
        grow_rate = 0.9
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K = 5
        current_K = 0
    elif current_K == 5:
        bootleneck_coeff = 0.1
        start, finish = np.random.randint(0,32)
        demography.add_population_parameters_change(limits[start], initial_size=bootleneck_coeff * initial_size)
        demography.add_population_parameters_change(limits[finish], initial_size=1. * initial_size)
        current_K = 0
    
    return demography


def get_set1(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':

    global current_K
    points = limits
    population = 10_000
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)   

    #current_K = np.random.choice([0,1,2,3,4])

    if current_K == 0: # const
        current_K = 1
    elif current_K == 1: # hl
        demography.add_population_parameters_change(2900, initial_size=0.5 * initial_size)
        demography.add_population_parameters_change(6000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(15000, initial_size=.5 * initial_size)
        demography.add_population_parameters_change(18000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(30000, initial_size=2. * initial_size)

        current_K = 2
    elif current_K == 2: # random
        a, b = 0.1, 2.0    
        for (i, point) in enumerate(points):
            coeff = np.random.uniform(a, b)
            if i != 0: 
                demography.add_population_parameters_change(point, initial_size=coeff * population)
        current_K = 0
    return demography

def get_set2(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':

    global current_K
    points = limits
    population = 10_000
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)   

    #current_K = np.random.choice([0,1,2,3,4])

    if current_K == 0:
        current_K = 1
    elif current_K == 1:
        demography.add_population_parameters_change(2900, initial_size=0.5 * initial_size)
        demography.add_population_parameters_change(6000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(15000, initial_size=.5 * initial_size)
        demography.add_population_parameters_change(18000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(30000, initial_size=2. * initial_size)

        current_K = 2
    elif current_K == 2:
        a, b = 0.1, 2.0    
        for (i, point) in enumerate(points):
            coeff = np.random.uniform(a, b)
            if i != 0: 
                demography.add_population_parameters_change(point, initial_size=coeff * population)
        current_K = 3
    elif current_K == 3:
        grow_rate = 1.1
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K = 4
    elif current_K == 4:
        grow_rate = 0.9
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K = 0
    
    return demography

def get_set2_plus(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':

    global current_K_plus
    points = limits
    population = 10_000
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)   

    #current_K_plus = np.random.choice([0,1,2,3,4,5])

    if current_K_plus == 0:
        current_K_plus = 1
    elif current_K_plus == 1:
        demography.add_population_parameters_change(2900, initial_size=0.5 * initial_size)
        demography.add_population_parameters_change(6000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(15000, initial_size=.5 * initial_size)
        demography.add_population_parameters_change(18000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(30000, initial_size=2. * initial_size)

        current_K_plus = 2
    elif current_K_plus == 2:
        a, b = 0.1, 2.0    
        for (i, point) in enumerate(points):
            coeff = np.random.uniform(a, b)
            if i != 0: 
                demography.add_population_parameters_change(point, initial_size=coeff * population)
        current_K_plus = 3
    elif current_K_plus == 3:
        grow_rate = 1.1
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K_plus = 4
    elif current_K_plus == 4:
        grow_rate = 0.9
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K_plus = 5
    elif current_K_plus == 5:
        demography.add_population_parameters_change(time=400,   population="A", initial_size=0.1 * initial_size)
        demography.add_population_parameters_change(time=2400,  population="A", initial_size=1.0 * initial_size)
        demography.add_population_parameters_change(time=8000,  population="A", initial_size=0.5 * initial_size)
        demography.add_population_parameters_change(time=40000, population="A", initial_size=1.0 * initial_size)
        demography.add_population_parameters_change(time=80000, population="A", initial_size=2.0 * initial_size)

        current_K_plus = 0
    
    return demography

def get_set3(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':

    global current_K
    points = limits
    population = 10_000
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)   

    #current_K = np.random.choice([0,1,2,3,4])

    if current_K == 0:
        current_K = 1
    elif current_K == 1:
        demography.add_population_parameters_change(2900, initial_size=0.5 * initial_size)
        demography.add_population_parameters_change(6000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(15000, initial_size=.5 * initial_size)
        demography.add_population_parameters_change(18000, initial_size=1. * initial_size)
        demography.add_population_parameters_change(30000, initial_size=2. * initial_size)

        # demography.add_population_parameters_change(time=400,   population="A", initial_size=0.1 * initial_size)
        # demography.add_population_parameters_change(time=2400,  population="A", initial_size=1.0 * initial_size)
        # demography.add_population_parameters_change(time=8000,  population="A", initial_size=0.5 * initial_size)
        # demography.add_population_parameters_change(time=40000, population="A", initial_size=1.0 * initial_size)
        # demography.add_population_parameters_change(time=80000, population="A", initial_size=2.0 * initial_size)

        current_K = 2
    elif current_K == 2:
        a, b = 0.1, 2.0    
        for (i, point) in enumerate(points):
            coeff = np.random.uniform(a, b)
            if i != 0: 
                demography.add_population_parameters_change(point, initial_size=coeff * population)
        current_K = 3
    elif current_K == 3:
        grow_rate = 1.1
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K = 4
    elif current_K == 4:
        grow_rate = 0.9
        cur_population = population
        for (i, point) in enumerate(points):
            if i != 0: 
                cur_population = cur_population * grow_rate
                demography.add_population_parameters_change(point, initial_size=cur_population)
        current_K = 5
        # current_K = 0
    
    elif current_K == 5: 
        demography.add_population_parameters_change(1500, initial_size=initial_size*0.8)
        demography.add_population_parameters_change(2200, initial_size=initial_size*0.6)
        demography.add_population_parameters_change(3000, initial_size=initial_size*0.4)

        demography.add_population_parameters_change(3400, initial_size=initial_size*0.6)
        demography.add_population_parameters_change(3800, initial_size=initial_size*0.8)
        demography.add_population_parameters_change(4200, initial_size=initial_size*1.0)


        demography.add_population_parameters_change(9800, initial_size=initial_size*1.25)
        demography.add_population_parameters_change(11000, initial_size=initial_size*1.5)
        demography.add_population_parameters_change(12500, initial_size=initial_size*2.0)

        demography.add_population_parameters_change(14000, initial_size=initial_size*1.5)
        demography.add_population_parameters_change(15500, initial_size=initial_size*1.25)
        demography.add_population_parameters_change(16500, initial_size=initial_size*1.0)    

        demography.add_population_parameters_change(21000, initial_size=initial_size*0.8)
        demography.add_population_parameters_change(24000, initial_size=initial_size*0.6)
        demography.add_population_parameters_change(27000, initial_size=initial_size*0.4)
        current_K = 0
        
    elif current_K == 42:
        bootleneck_coeff = 0.1
        start, finish = np.random.randint(0,32)
        demography.add_population_parameters_change(limits[start], initial_size=bootleneck_coeff * initial_size)
        demography.add_population_parameters_change(limits[finish], initial_size=1. * initial_size)
        current_K = 0
    
    return demography


def get_(initial_size: int = 10_000, random_seed: int = 42) -> 'msprime.Demography':
    
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)   

    start = np.random.randint(0,31)
    finish = start + 1
    coeff = np.random.choice([0.5, 2])

    demography.add_population_parameters_change(limits[start], initial_size=coeff * initial_size)
    demography.add_population_parameters_change(limits[finish], initial_size=1. * initial_size)
    return  demography


def get_strong_bootleneck(initial_size: int = 10_000, random_seed: int = 42, bootleneck_coeff = 0.1) -> 'msprime.Demography':
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)
    demography.add_population_parameters_change(2900, initial_size=bootleneck_coeff * initial_size)
    demography.add_population_parameters_change(6000, initial_size=1. * initial_size)

    return demography


def get_strong_antibootleneck(initial_size: int = 10_000, random_seed: int = 42, bootleneck_coeff = 10) -> 'msprime.Demography':
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=initial_size)
    demography.add_population_parameters_change(2900, initial_size=bootleneck_coeff * initial_size)
    demography.add_population_parameters_change(6000, initial_size=1. * initial_size)

    return demography



def simple_split(time: float, N: int, split_const: int = 5000) -> int:
    return int(min(time // split_const, N - 1))


def exponent_split(time: float, N: int) -> int:
    for i, limit in enumerate(limits):
        if limit > time:
            return max(i - 1, 0)
    return max(len(limits) - 1, 0)


def non_filter(_input):
    return _input


def do_filter(mutations, l=L_HUMAN):
    l = int(len(mutations) / 100)
    genome = [0] * l
    for j in range(int(len(mutations) / 100)):
        genome[j] = max(mutations[j * 100:(j + 1) * 100])
    return genome


def do_filter_2(d_times, l=L_HUMAN):
    genome = [0] * int(len(d_times) / 100)
    for j in range(int(len(d_times) / 100)):
        genome[j] = stats.mode(d_times[j * 100:(j + 1) * 100]).mode[0]
    return genome


def generate_ms_command(random_seed=None):
    if random_seed is not None:
        rnd.seed(random_seed)
    # Здесь мы один раз выбираем точки, в которых можно менять Ne
    # -eN 0.0 3.0 -eN 0.025 0.2 -eN 0.175 1.5 -eN 3 3 -eN 10.0 3
    times = [0.0, 0.025, 0.175, 3.0, 10.0]
    eps = 1e-5
    times[0] = eps
    T = []
    for i in range(4):
        T.append(times[i])
        T.append((times[i] ** 2 * times[i + 1]) ** (1. / 3.))
        T.append((times[i] * times[i + 1] ** 2) ** (1. / 3.))
    T.append(10.0)
    T[0] = 0.0

    # Эта функция генерирует траектории (сильно ad hoc, но на вид получается неплохо, мне кажется)
    def GenerateNE(T):
        Ne = [0.0 for i in range(len(T))]

        # Time interval 0.0 - 0.025
        N1 = rnd.uniform(0.5, 5)
        Ne[0] = N1
        Ne[1] = N1
        Ne[2] = N1

        N2 = exp(rnd.uniform(log(0.1 + 0.2), log(5.0 + 0.2))) - 0.2
        N3 = rnd.uniform(0.5, 5)
        N4 = rnd.uniform(0.5, 5)

        # Time interval 0.025 - 0.175
        shift = Ne[2] - 0.3
        shift = 0.0
        if shift >= 0.0:
            Ne[3] = exp(rnd.uniform(log(N1 + shift), log(N2 + shift))) - shift
        else:
            Ne[3] = rnd.uniform(N1, N2)
        Ne[4] = N2
        if shift >= 0.0:
            Ne[5] = exp(rnd.uniform(log(N2 + shift), (log(N3 + shift) + log(N2 + shift)) / 2.0)) - shift
        else:
            Ne[5] = rnd.uniform(N2, (N3 + N2) / 2.0)

        # Time interval 0.175 - 3.0
        Ne[6] = rnd.uniform((N3 + N2) / 2.0, N3)
        Ne[7] = N3
        Ne[8] = rnd.uniform(N3, N4)

        # Time interval 3.0 - 10.0
        Ne[9] = N4
        Ne[10] = rnd.uniform(0.5, 5)
        Ne[11] = rnd.uniform(0.5, 5)
        Ne[12] = 1.0
        return (Ne)

    anwser = ''
    Ne = GenerateNE(T)
    # Собираем строку в формате ms
    for t, ne in zip(T, Ne):
        anwser += f"-eN {t} {ne} "
    return anwser


def get_demographcs_from_ms_command(ms=None, init_population: int = 10_000, random_seed=42) -> 'msprime.Demography':
    if type(ms) != str:
        if ms is not None:
            ms = ms(random_seed)
        else:
            ms = generate_ms_command(random_seed)

    graph = demes.from_ms(ms, N0=init_population, deme_names=["A"])
    demography = msprime.Demography.from_demes(graph)
    return demography


class DataGenerator():
    def __init__(self,
                 recombination_rate: float = RHO_HUMAN,
                 mutation_rate: float = MU_HUMAN,
                 demographic_events: list = None,
                 population: int = None,
                 number_intervals: int = N,
                 splitter=exponent_split,  # maust be annotiede
                 num_replicates: int = 1,
                 lengt: int = L_HUMAN,
                 model: str = "hudson",
                 random_seed: int = 42,
                 sample_size: int = 1,
                 return_local_times: bool = True,
                 return_full_dist: bool = True,
                 genome_postproccessor=non_filter,
                 times_postproccessor=non_filter,
                 ):

        self.sample_size = sample_size
        self.recombination_rate = recombination_rate
        self.mutation_rate = mutation_rate
        self.num_replicates = num_replicates
        if not demographic_events:
            if not population:
                raise BaseException(
                    "Eiter demographic_events or population must be speciefied")
            demographic_events = msprime.Demography()
            demographic_events.add_population(
                name="A", initial_size=population)
        self.demographic_events = demographic_events
        self.splitter = splitter
        self.model = model
        self.len = lengt
        self.random_seed = random_seed
        self.number_intervals = number_intervals
        self._data = None
        self.genome_postproccessor = genome_postproccessor
        self.times_postproccessor = times_postproccessor

        self.return_local_times = return_local_times
        self.return_full_dist = return_full_dist

    def run_simulation(self):
        """
        return generator(tskit.TreeSequence)
        function run the simulation with given parametrs
        """
        self._data = msprime.sim_ancestry(
            recombination_rate=self.recombination_rate,
            sequence_length=self.len,
            num_replicates=self.num_replicates,
            demography=self.demographic_events,
            model=self.model,
            random_seed=self.random_seed,
            samples=self.sample_size)
        return self._data

    def __iter__(self):
        return self

    def __next__(self):
        """
        return haplotype, recombination points and coalescent time
        """
        if self._data is None:
            self.run_simulation()

        try:
            tree = next(self._data)
        except StopIteration:
            raise StopIteration

        mutated_ts = msprime.sim_mutations(
            tree, rate=self.mutation_rate)  # random_seed

        # times = [0]*self.len
        d_times = [0] * self.len
        mutations = [0] * self.len
        prior_dist = [0.0] * self.number_intervals

        # for m in mutated_ts.mutations():

        for m in mutated_ts.tables.sites.position:
            mutations[int(m)] = 1

        for t in mutated_ts.aslist():
            interval = t.get_interval()
            left = interval.left
            right = interval.right
            time = t.get_total_branch_length() / 2
            # times[int(left):int(right)] = [time]*int(right-left)
            d_times[int(left):int(right)] = [self.splitter(
                time, self.number_intervals)] * int(right - left)
            prior_dist[self.splitter(
                time, self.number_intervals)] += (int(right - left)) / self.len

        if self.return_local_times and self.return_full_dist:
            return self.genome_postproccessor(mutations), prior_dist, self.times_postproccessor(d_times)
        elif self.return_local_times and not self.return_full_dist:
            return self.genome_postproccessor(mutations), self.times_postproccessor(d_times)
        elif self.return_full_dist and not self.return_local_times:
            return self.genome_postproccessor(mutations), prior_dist
        return None


# get_const_demographcs
# get_test_demographcs
# get_demographcs_from_ms_command
def get_generator(num_genomes: int,
                  genome_length: int,
                  num_generators: int = 1,
                  random_seed: int = 42,
                  genome_postproccessor=non_filter,
                  times_postproccessor=non_filter
                  ) -> 'Generator[DataGenerator]':
    yield from [DataGenerator(lengt=genome_length,
                              num_replicates=num_genomes,
                              demographic_events=get_const_demographcs(),
                              random_seed=random_seed + i,
                              genome_postproccessor=genome_postproccessor,
                              times_postproccessor=times_postproccessor
                              ) for i in range(num_generators)]


def get_list(num_genomes: int,
             genome_length: int,
             num_generators: int = 1,
             random_seed: int = 42,
             genome_postproccessor=non_filter,
             times_postproccessor=non_filter
             ) -> 'Tuple(List[int], List[int], List[int])':
    generator = get_generator(num_genomes=num_genomes,
                              genome_length=genome_length,
                              num_generators=num_generators,
                              random_seed=random_seed,
                              genome_postproccessor=genome_postproccessor,
                              times_postproccessor=times_postproccessor
                              )
    genomes = []
    d_times = []
    prior_dist = []
    for it in generator:
        for g, t, d in it:
            genomes.append(g)
            d_times.append(t)
            prior_dist.append(d)

    return shuffle(genomes, d_times, prior_dist, random_state=random_seed)


def get_liner_generator(num_genomes: int,
                        genome_length: int,
                        num_generators: int = 1,
                        random_seed: int = 42,
                        return_local_times: bool = True,
                        return_full_dist: bool = True,
                        genome_postproccessor=non_filter,
                        times_postproccessor=non_filter,
                        demographic_events_generator=generate_demographic_events
                        ) -> 'Generator':
    generators = [DataGenerator(num_replicates=num_genomes,
                                lengt=genome_length,
                                demographic_events=demographic_events_generator(random_seed=random_seed + 100 * i),
                                #random_seed=random_seed + i,
                                return_local_times=return_local_times,
                                return_full_dist=return_full_dist,
                                genome_postproccessor=genome_postproccessor,
                                times_postproccessor=times_postproccessor
                                ) for i in range(num_generators)]
    generators = shuffle(generators, random_state=random_seed)

    while generators:
        i = np.random.choice(len(generators))
        g = generators[i]
        try:
            # Try to yield genome
            yield next(g)
        except StopIteration:
            # If msprime generator stops, pop it
            generators.pop(i)


###
initial_size = 10_000
T = 2 * 2 * 10_000

demography_1 = msprime.Demography()
demography_1.add_population(name="A", initial_size=4.5 * initial_size)
demography_1.add_population_parameters_change(0.025 * T, initial_size=0.2 * initial_size)
demography_1.add_population_parameters_change(0.175 * T, initial_size=3 * initial_size)
demography_1.add_population_parameters_change(0.625 * T, initial_size=1.8 * initial_size)
demography_1.add_population_parameters_change(3 * T, initial_size=3.2 * initial_size)
demography_1.add_population_parameters_change(8 * T, initial_size=5.5 * initial_size)

demography_2 = msprime.Demography()
demography_2.add_population(name="A", initial_size=1.5 * initial_size)
demography_2.add_population_parameters_change(3.2 * T, initial_size=3 * initial_size)

demography_3 = msprime.Demography()
demography_3.add_population(name="A", initial_size=3.0 * initial_size)
demography_3.add_population_parameters_change(0.05 * T, initial_size=0.25 * initial_size)
demography_3.add_population_parameters_change(0.18 * T, initial_size=1.5 * initial_size)
demography_3.add_population_parameters_change(0.32 * T, initial_size=3 * initial_size)


###

def get_generator_spetial_demography(demography,
                                     num_genomes: int = 100,
                                     genome_length: int = L_HUMAN,
                                     num_generators: int = 1,
                                     random_seed: int = 42) -> 'Generator[DataGenerator]':
    yield from [DataGenerator(lengt=genome_length,
                              num_replicates=num_genomes,
                              demographic_events=demography,
                              random_seed=random_seed,
                              )]


if __name__ == "__main__":

    num_model = int(sys.argv[6])
    x_path = os.path.join(sys.argv[1], "x")
    y_path = os.path.join(sys.argv[1], "y")
    pd_path = os.path.join(sys.argv[1], "PD")

    name = 0
    print(f'Path: {sys.argv[1]}')
    print(f"Num_model: {num_model}")
    print(f"Num replicates: {sys.argv[2]}")
    for j in range(num_model):
        generator = DataGenerator(
            demographic_events=generate_demographic_events_complex(),
            splitter=exponent_split,
            num_replicates=int(sys.argv[2])
        )
        generator.run_simulation()
        # return mutations, times, None, prior_dist, None
        for x, y, t in generator:
            print(name)
            x = np.array(x, dtype=np.int64)
            y = np.array(y)
            pd = np.array(t)

            np.save(x_path + "/" + str(name), x)
            np.save(y_path + "/" + str(name), y)
            # np.save(pd_path + "/" + str(name), pd)
            name += 1

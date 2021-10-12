import os
import sys
from math import (exp, log)

import msprime
from sklearn.utils import shuffle
import numpy as np

N = 32  # int(sys.argv[4])

T_max = 20  # 400_000
coeff = np.log(T_max) / (N - 1)

limits = [np.exp(coeff * i) for i in range(N)]
limits = [2_000 * (np.exp(i * np.log(1 + 10 * T_max) / N) - 1)
          for i in range(N)]
limits = [2000.] + [l for l in limits if l > 2000.0]

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

L_HUMAN = 30_000_000  # int(float(sys.argv[3])) #30_000_000
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


def generate_demographic_events_complex(population: int = None) -> 'msprime.Demography':
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


def generate_demographic_events(population: int = None) -> 'msprime.Demography':
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


def simple_split(time: float, N: int, split_const: int = 5000) -> int:
    return int(min(time // split_const, N - 1))


def exponent_split(time: float, N: int) -> int:
    for i, limit in enumerate(limits):
        if limit > time:
            return i
    return len(limits) - 1


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
        
        for m in mutated_ts.mutations():
            mutations[int(m.position)] = 1
        
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
        
        return mutations, d_times #, prior_dist


def get_generator(num_genomes: int,
                  genome_length: int,
                  num_generators: int = 1,
                  random_seed: int = 42) -> 'Generator[DataGenerator]':
    yield from [DataGenerator(lengt=genome_length,
                              num_replicates=num_genomes,
                              demographic_events=generate_demographic_events(),
                              random_seed=random_seed + i,
                              ) for i in range(num_generators)]


def get_list(num_genomes: int,
             genome_length: int,
             num_generators: int = 1,
             random_seed: int = 42) -> 'Tuple(List[int], List[int], List[int])':
    generator = get_generator(num_genomes=num_genomes,
                              genome_length=genome_length,
                              num_generators=num_generators,
                              random_seed=random_seed,
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
                        random_seed: int = 42) -> 'Generator':
    generators = [DataGenerator(num_replicates=num_genomes,
                                lengt=genome_length,
                                demographic_events=generate_demographic_events(),
                                random_seed=random_seed + i,
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
            splitter=simple_split,
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

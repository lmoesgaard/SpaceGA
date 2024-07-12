import os
import random
from joblib import Parallel, delayed
import time
import warnings

from rdkit import Chem, rdBase
import pandas as pd

from ga.crossover import crossover
from ga.mutation import mutate
from ga.molecule import synthesizeable
from ga.neutralize import neutralize_molecules
from filtering.filter import Filtering
from utils import smifile2df, sim_filter, get_scorer
from setup import read_config, make_workdir

warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor",
                        category=UserWarning)

class GA:
    def __init__(self, json_file):
        settings = read_config(json_file)
        make_workdir(settings)
        self.O: bool = settings["O"]
        self.o: str = settings["o"]
        self.i: str = settings["i"]
        self.p_size: int = settings["p_size"]
        self.children: int = settings["children"]
        self.generation_size = int(self.p_size * self.children)
        self.crossover_rate: float = settings["crossover_rate"]
        self.mutation_rate: float = settings["mutation_rate"]
        self.iterations: int = settings["iterations"]
        self.cpu: int = settings["cpu"]
        self.gpu: int = settings["gpu"]
        self.sim_cutoff: float = settings["sim_cutoff"]
        self.scorer = get_scorer(settings["mode"], settings["scoring_inputs"])
        self.filter = Filtering(settings["filtering_inputs"])
        self.gen = 1
        self.time = time.time()
        self.population = None
        self.taken = set()
        self.start_population()

    def start_population(self):
        population = smifile2df(self.i).sample(self.generation_size)
        population = population.drop("name", axis=1)
        population["mol"] = population.apply(lambda x: Chem.MolFromSmiles(x.smi), axis=1)
        population["scores"] = self.scorer.score(population.smi, self.cpu, self.gpu)
        population["generation"] = self.gen
        self.update_pop(population)
        self.write_status()

    def pick_random_item(self, n, col):
        weights = self.population["scores"] - min(self.population["scores"])
        sel = self.population.sample(n, weights=weights)
        return list(sel[col])

    def generate_molecule(self):
        good_mol = False
        rdBase.DisableLog("rdApp.error")
        do_cross = random.random() < self.crossover_rate
        do_mutate = random.random() < self.mutation_rate
        count = 0
        while not good_mol:
            count += 1
            if do_cross:
                parents = self.pick_random_item(2, "mol")
                mol = crossover(parents[0], parents[1])
            else:
                mol = self.pick_random_item(1, "mol")[0]
            if mol is not None:
                if do_mutate:
                    mol = mutate(mol)
                if mol is not None:
                    mol = neutralize_molecules([mol])[0]
                    if mol is not None:
                        if self.filter.filter_mol(mol):
                            good_mol = synthesizeable(mol)
        return Chem.MolFromSmiles(Chem.MolToSmiles(mol)) # ensure cannonical

    def mate(self):
        target = int((self.generation_size * 5) / self.cpu)  # sample 5x more molecules to compensate for duplicates
        return [self.generate_molecule() for _ in range(target)]

    def reproduce(self):
        self.gen += 1
        children = Parallel(n_jobs=self.cpu)(delayed(self.mate)() for _ in range(self.cpu))
        children = {Chem.MolToSmiles(mol) for mol_list in children for mol in mol_list}
        children = [child for child in children if child not in self.taken]
        if len(children) > self.generation_size:
            children = random.sample(children, self.generation_size)
        mol_lst = [Chem.MolFromSmiles(mol) for mol in children]
        offspring = pd.DataFrame({"smi": children, "mol": mol_lst})
        offspring["scores"] = self.scorer.score(offspring.smi, self.cpu, self.gpu)
        offspring["generation"] = self.gen
        outname = os.path.join(self.o, f"gen_{self.gen}.parquet")
        offspring[["smi", "scores", "generation"]].to_parquet(outname)
        return offspring

    def write_status(self):
        seconds = time.time() - self.time
        timepoint = time.strftime('%H:%M:%S', time.gmtime(seconds))
        print(f"{timepoint}: Generation {self.gen} - {self.scorer.count} molecules have been scored")

    def update_pop(self, population):
        outname = os.path.join(self.o, f"gen_{self.gen}.parquet")
        population[["smi", "scores", "generation"]].to_parquet(outname)
        self.taken.update(list(population.smi))
        population = population.sort_values("scores", ascending=False)
        mask = sim_filter(population.mol.to_list(), self.p_size, self.sim_cutoff)
        population = population[mask]
        self.population = population
        outname = os.path.join(self.o, f"pop_{self.gen}.parquet")
        self.population[["smi", "scores", "generation"]].to_parquet(outname)

    def run(self):
        for _ in range(1, self.iterations):
            offspring = self.reproduce()
            population = pd.concat([self.population, offspring])
            self.update_pop(population)
            self.write_status()

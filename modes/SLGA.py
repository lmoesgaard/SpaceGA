import os
import random
from joblib import Parallel, delayed
import time

from rdkit import Chem, rdBase
import pandas as pd

from ga.crossover import crossover
from ga.neutralize import neutralize_molecules
from filtering.filter import Filtering
from utils import smifile2df, sim_filter, get_scorer
from setup import read_config, make_workdir
from sl.spacelight import simsearch


class SLGA:
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
        self.iterations: int = settings["iterations"]
        self.cpu: int = settings["cpu"]
        self.gpu: int = settings["gpu"]
        self.sim_cutoff = settings["sim_cutoff"]
        self.space = settings["space"]
        self.scorer = get_scorer(settings["mode"], settings["scoring_inputs"])
        self.filter = Filtering(settings["filtering_inputs"])
        self.gen = 1
        self.time = time.time()
        self.population = self.start_population()
        self.scratch = self.make_scratch()
        self.search = simsearch(self.scratch, self.space, self.filter, self.children, settings["spacelight"])

    def make_scratch(self):
        scratch = os.path.join(self.o, "scratch")
        try:
            os.mkdir(scratch)
        except:
            pass
        return scratch

    def start_population(self):
        population = smifile2df(self.i).sample(self.generation_size)
        population["mol"] = population.apply(lambda x: Chem.MolFromSmiles(x.smi), axis=1)
        population["scores"] = self.scorer.score(population.smi, self.cpu, self.gpu)
        population["generation"] = self.gen
        outname = os.path.join(self.o, f"gen_{self.gen}.parquet")
        population[["smi", "name", "scores", "generation"]].to_parquet(outname)
        return population

    def pick_random_item(self, n, col):
        weights = self.population["scores"] - min(self.population["scores"])
        sel = self.population.sample(n, weights=weights)
        return list(sel[col])

    def generate_molecule(self):
        good_mol = False
        rdBase.DisableLog("rdApp.error")
        do_cross = random.random() < self.crossover_rate
        while not good_mol:
            if do_cross:
                parents = self.pick_random_item(2, "mol")
                mol = crossover(parents[0], parents[1])
            else:
                mol = self.pick_random_item(1, "mol")[0]
            if mol is not None:
                mol = neutralize_molecules([mol])[0]
                if mol is not None:
                    if self.filter.filter_mol(mol):
                        smi = Chem.MolToSmiles(mol)
                        if do_cross:
                            children = self.search.search(smi, drop_first=False, scrample=False)
                        else:
                            children = self.search.search(smi)
                        if children is not None:
                            good_mol = True
        return children

    def reproduce(self):
        self.gen += 1
        offspring = Parallel(n_jobs=self.cpu)(delayed(self.generate_molecule)() for _ in range(self.p_size))
        offspring = pd.concat(offspring)
        self.search.taken.update(offspring["name"])
        offspring["scores"] = self.scorer.score(offspring.smi, self.cpu, self.gpu)
        offspring["generation"] = self.gen
        outname = os.path.join(self.o, f"gen_{self.gen}.parquet")
        offspring[["smi", "name", "scores", "generation"]].to_parquet(outname)
        return offspring

    def write_status(self):
        seconds = time.time() - self.time
        timepoint = time.strftime('%H:%M:%S', time.gmtime(seconds))
        print(f"{timepoint}: Generation {self.gen} - {self.scorer.count} molecules have been scored")
        outname = os.path.join(self.o, f"pop_{self.gen}.parquet")
        self.population[["smi", "name", "scores", "generation"]].to_parquet(outname)

    def run(self):
        self.write_status()
        for _ in range(1, self.iterations):
            offspring = self.reproduce()
            population = pd.concat([self.population, offspring])
            population = population.sort_values("scores", ascending=False)
            mask = sim_filter(population.mol.to_list(), self.p_size, self.sim_cutoff)
            population = population[mask]
            self.population = population
            self.write_status()
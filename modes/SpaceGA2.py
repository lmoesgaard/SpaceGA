import os
import random
from joblib import Parallel, delayed
import time
import warnings
import math
import sys

from rdkit import Chem, rdBase
import pandas as pd
import numpy as np
import torch

from ga.crossover import crossover
from ga.neutralize import neutralize_molecules
from filtering.filter import Filtering
from utils import smifile2df, sim_filter, get_scorer, split_array, get_train_mask, smi2array, split_df
from setup import read_config, make_workdir
from sl.spacelight import simsearch
from ml.train import train_model, get_model
from ml.data import DataGen
from ml.pred import PredictScores

warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor",
                        category=UserWarning)

class SpaceGA2:
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
        self.sl_cpu: int = settings["cpu"]
        if "sl_cpu" in settings:
            self.sl_cpu: int = settings["sl_cpu"]
        self.gpu: int = settings["gpu"]
        self.sim_cutoff = settings["sim_cutoff"]
        self.space = settings["space"]
        self.f_comp = settings["f_comp"]
        self.bsize: int = settings["bsize"]
        self.model_name = settings["model_name"]
        self.model_path = self.outname("model.pth")
        settings["scoring_inputs"]["workdir"] = settings["o"]
        self.scorer = get_scorer(settings["mode"], settings["scoring_inputs"])
        self.filter = Filtering(settings["filtering_inputs"])
        self.gen = 1
        self.time = time.time()
        self.scratch = self.make_ml_dirs()
        self.population = self.start_population()
        self.search = simsearch(self.scratch, self.space, self.filter, self.children, self.f_comp, settings["spacelight"], al=True)

    def outname(self, name):
        return os.path.join(self.o, name)

    def make_ml_dirs(self):
        for name in ["train", "val", "scratch"]:
            try:
                os.mkdir(self.outname(name))
            except:
                pass
        return self.outname("scratch")
    
    def start_population(self):
        population = smifile2df(self.i).sample(self.generation_size)
        population["mol"] = population.apply(lambda x: Chem.MolFromSmiles(x.smi), axis=1)
        population["scores"] = self.scorer.score(population.smi, self.cpu, self.gpu)
        population["generation"] = self.gen
        self.save_data(population)
        return self.update_pop(population)

    def update_pop(self, population):
        outname = os.path.join(self.o, f"gen_{self.gen}.parquet")
        population[["smi", "name", "scores", "generation"]].to_parquet(outname)
        population = population.sort_values("scores", ascending=False)
        mask = sim_filter(population.mol.to_list(), self.p_size, self.sim_cutoff)
        population = population[mask]
        outname = os.path.join(self.o, f"pop_{self.gen}.parquet")
        population[["smi", "name", "scores", "generation"]].to_parquet(outname)
        self.write_status(population)
        return population

    def pick_random_item(self, n, col):
        weights = self.population["scores"] - min(self.population["scores"])
        sel = self.population.sample(n, weights=weights)
        return list(sel[col])

    def generate_molecule(self, n_tries=10):
        good_mol = False
        rdBase.DisableLog("rdApp.error")
        do_cross = random.random() < self.crossover_rate
        while not good_mol and n_tries != 0:
            n_tries -= 1
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
        if not good_mol:
            return None
        else:
            return children

    def save_data(self, df, max_len=100000):
        y = df.scores.to_numpy()
        x = Parallel(n_jobs=self.cpu)(delayed(smi2array)(smi) for smi in df.smi)
        x = np.array(x).astype(int)
        if self.gen == 1:
            masks = get_train_mask(y, [0.8, 0.2, 0.0])
        else:
            masks = {"train": np.ones(len(y)).astype(bool)}
        for x_or_y, data in zip(["x", "y"], [x, y]):
            for splitname in masks:
                a = data[masks[splitname]]
                if len(a) > 0:
                    n_subsets = math.ceil(len(a) / max_len)
                    for n, sub_a in enumerate(split_array(a, n_subsets)):
                        name = os.path.join(self.outname(splitname), f"{self.gen}{n}_{x_or_y}.npz")
                        np.savez_compressed(name, data_array=sub_a)

    def train(self):
        device = torch.device("cuda:0")
        dataset = DataGen(self.outname("train"), self.outname("val"), device=device, batch_size=self.bsize)
        coach = train_model(self.model_name, self.model_path, dataset, device)
        coach.train_model()
        del coach, dataset

    def predict(self, smi):
        screener = PredictScores(self.model_name, self.model_path, self.cpu, self.gpu)
        output = screener.run_pred(smi)
        return np.array(output)

    def al(self, offspring):
        self.train()
        screener = PredictScores(self.model_name, self.model_path, self.cpu, self.gpu)
        predictions = []
        for subset in split_df(offspring, math.ceil(offspring.shape[0]/100000)):
            predictions.append(screener.run_pred(subset.smi))
        offspring["predictions"] = np.concatenate(predictions)
        offspring = offspring.sort_values("predictions", ascending=False)
        offspring = offspring.drop("predictions", axis=1)
        return offspring.head(self.generation_size)

    def reproduce(self):
        self.gen += 1
        offspring = Parallel(n_jobs=self.sl_cpu)(delayed(self.generate_molecule)() for _ in range(self.p_size))
        try:
            offspring = pd.concat(offspring)
        except:
            print("Failed to reproduce")
            sys.exit()
        offspring = offspring.drop_duplicates("name")
        offspring = self.al(offspring)
        self.search.taken.update(offspring["name"])
        offspring["scores"] = self.scorer.score(offspring.smi, self.cpu, self.gpu)
        offspring["generation"] = self.gen
        return offspring

    def write_status(self, population):
        seconds = time.time() - self.time
        timepoint = time.strftime('%H:%M:%S', time.gmtime(seconds))
        scores = population.scores
        message = f"{timepoint}: Generation {self.gen} - {self.scorer.count} molecules have been scored"
        message += f" (mean: {scores.mean():.2f}, best: {scores.max():.2f})"
        print(message)

    def run(self):
        for _ in range(1, self.iterations):
            offspring = self.reproduce()
            self.save_data(offspring)
            population = pd.concat([self.population, offspring])
            self.population = self.update_pop(population)

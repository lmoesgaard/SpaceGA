from dataclasses import dataclass, field, asdict
from typing import Dict, Optional
from rdkit import Chem
import time
import os
import json
import pandas as pd
import os
import shutil
import logging
import time

from SpaceGA.utils import sim_filter
from SpaceGA.ml.mltools import MLtools
from SpaceGA.ga.reproduce import Reproduce


@dataclass
class SpaceGASettings:
    # Required arguments
    O: bool
    o: str
    i: str
    space: str
    spacelight: str
    
    # Optional arguments
    p_size: int = 100
    children: int = 100
    crossover_rate: float = 0.2
    iterations: int = 10
    cpu: int = 1
    sl_cpu: Optional[int] = None
    al: bool = False
    sim_cutoff: float = 1.00
    f_comp: int = 100
    fp_type: str = "ECFP4"
    model_name: str = "NN1"
    config: str = 'json_files/spacega2.json'
    scoring_tool: Dict[str, float] = field(default_factory=lambda: 
                                           {"module": "scoring", "tool": "LogPSearch"})
    filtering_inputs: Dict[str, float] = field(default_factory=lambda: {})
    scoring_inputs: Dict[str, float] = field(default_factory=lambda: {})

    def __post_init__(self):
        if self.sl_cpu is None:
            self.sl_cpu = self.cpu

class SpaceGA(SpaceGASettings, MLtools, Reproduce):
    def __init__(self, generation, **kwargs):
        super().__init__(**kwargs)
        self.settings = asdict(self)

        if generation == 0:
            self.gen = generation
            self.make_dir(self.o)
            self.scratch = self.make_dir("scratch")
            self.log = self.make_dir("log")
            if self.al:
                self.train_dir = self.make_dir("train")
                self.setup_ml()
            self.generation_size = self.p_size * self.children
            self.population = self.search = None
            self.taken = []
            with open(os.path.join(self.log, f"state_r{generation}.json"), "w") as f:
                json.dump(self.__dict__, f)
        else:
            with open(os.path.join(self.o, "log", f"state_r{generation}.json"), "r") as f:
                state = json.load(f)
            self.__dict__.update(state)
            self.population = pd.read_pickle(os.path.join(self.o, f"pop_{generation}.pkl"))
            self.population["mol"] = self.population.smi.apply(lambda x: Chem.MolFromSmiles(x))
        self.time = time.time()

    def outname(self, name):
        return os.path.join(self.o, name)
    
    def make_dir(self, name):
        path = self.outname(name)
        try:
            os.mkdir(path)
        except FileExistsError:
            if self.O:
                shutil.rmtree(path)
                os.mkdir(path)
            else:
                logging.error(f"Failed to create directory {path}", exc_info=True)
        return path
    
    def write_status(self, population):
        seconds = time.time() - self.time
        timepoint = time.strftime('%H:%M:%S', time.gmtime(seconds))
        scores = population.scores
        message = f"{timepoint}: Generation {self.gen} (mean: {scores.mean():.2f}, best: {scores.max():.2f})"
        print(message)


    def update_pop(self, offspring):
        if self.al:
            offspring = self.update_model(offspring)
        outname = os.path.join(self.log, f"gen_{self.gen}.pkl")
        offspring[["smi", "name", "scores", "generation"]].to_pickle(outname)
        population = pd.concat([self.population, offspring])
        population = population.sort_values("scores", ascending=False)
        mask = sim_filter(population.mol.to_list(), self.p_size, self.sim_cutoff)
        population = population[mask]
        outname = os.path.join(self.o, f"pop_{self.gen}.pkl")
        population[["smi", "name", "scores", "generation"]].to_pickle(outname)
        self.write_status(population)
        self.population = None
        with open(os.path.join(self.log, f"state_r{self.gen}.json"), "w") as f:
            json.dump(self.__dict__, f)
        self.population = population
        return population

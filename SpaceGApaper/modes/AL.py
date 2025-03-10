import os
import random
from joblib import Parallel, delayed
import time
import math
from typing import List

import numpy as np
import pandas as pd
import torch

from utils import smi2array, get_train_mask, split_array, get_scorer
from ml.train import train_model
from ml.pred import MlScreen
from ml.data import DataGen
from setup import read_config, make_workdir


class AL:
    def __init__(self, json_file):
        settings = read_config(json_file)
        make_workdir(settings)
        self.O: bool = settings["O"]
        self.o: str = settings["o"]
        self.i: str = settings["i"]
        self.p_size: int = settings["p_size"]
        self.iterations: int = settings["iterations"]
        self.cpu: int = settings["cpu"]
        self.gpu: int = settings["gpu"]
        self.maxmodels: int = settings["maxmodels"]
        self.init_split: List[float, float, float] = settings["init_split"]
        self.bsize: int = settings["bsize"]
        self.model_name = settings["model_name"]
        self.model_path = self.outname("model.pth")
        settings["scoring_inputs"]["workdir"] = settings["o"]
        self.scorer = get_scorer(settings["mode"], settings["scoring_inputs"])
        self.gen = 1
        self.make_ml_dirs()
        self.file_prefixes = self.check_lib()
        self.time = time.time()
        self.taken = self.start_population()

    def outname(self, name):
        return os.path.join(self.o, name)

    def make_ml_dirs(self):
        for name in ["train", "val", "test", "scratch"]:
            try:
                os.mkdir(self.outname(name))
            except:
                pass

    def check_lib(self):
        smi_dir = os.listdir(os.path.join(self.i, "smis"))
        return [x.split(".")[0] for x in smi_dir if x.endswith("parquet")]

    def random_df(self):
        prefix = self.random_prefix()
        df = self.read_df(prefix)
        return df

    def random_prefix(self):
        prefix = random.choice(self.file_prefixes)
        return prefix

    def read_df(self, prefix):
        file = os.path.join(self.i, "smis", f"{prefix}.parquet")
        df = pd.read_parquet(file)
        df = df.reset_index(names="unique")
        df["unique"] = df["unique"].apply(lambda x: f"{prefix}_{x}")
        return df

    def start_population(self):
        df = self.random_df()
        while len(df) < self.p_size:
            df = pd.concat([df, self.random_df()])
            df = df.drop_duplicates("name")
        df = df.sample(self.p_size)
        df["scores"] = self.scorer.score(df.smi, self.cpu, self.gpu)
        df["generation"] = self.gen
        self.save_data(df)
        taken = set(df.unique.unique())
        return taken

    def save_data(self, df, max_len=100000):
        df[["smi", "name", "scores", "generation"]].to_parquet(self.outname(f"gen_{self.gen}.parquet"))
        y = df.scores.to_numpy()
        x = Parallel(n_jobs=self.cpu)(delayed(smi2array)(smi) for smi in df.smi)
        x = np.array(x).astype(int)
        if self.gen == 1:
            masks = get_train_mask(y, self.init_split)
        else:
            masks = {"train": np.ones(len(y)).astype(bool)}
        for x_or_y, data in zip(["x", "y"], [x, y]):
            for splitname in masks:
                a = data[masks[splitname]]
                n_subsets = math.ceil(len(a) / max_len)
                for n, sub_a in enumerate(split_array(a, n_subsets)):
                    name = os.path.join(self.outname(splitname), f"{self.gen}{n}_{x_or_y}.npz")
                    np.savez_compressed(name, data_array=sub_a)

    def train(self):
        self.gen += 1
        device = torch.device("cuda:0")
        dataset = DataGen(self.outname("train"), self.outname("val"), device=device, batch_size=self.bsize)
        coach = train_model(self.model_name, self.model_path, dataset, device)
        coach.train_model()
        del coach, dataset

    def predict(self):
        screener = MlScreen(self.model_name, self.model_path, self.cpu, self.gpu, self.file_prefixes, self.p_size, self.i, self.maxmodels)
        output = screener.pred_parallel()
        output = pd.concat(output)
        output = output[output.unique.apply(lambda x: x not in self.taken)]
        output = output.sort_values("pred", ascending=False).head(self.p_size)
        return output

    def extract(self, prefix, subset):
        rows = [int(x.split("_")[1]) for x in subset.unique]
        return self.read_df(prefix).iloc[rows]

    def extract_best(self, df):
        output = Parallel(n_jobs=self.cpu)(
            delayed(self.extract)(prefix, subset) for prefix, subset in df.groupby("prefix"))
        return pd.concat(output)

    def write_status(self):
        seconds = time.time()-self.time
        timepoint = time.strftime('%H:%M:%S', time.gmtime(seconds))
        print(f"{timepoint}: Generation {self.gen} - {self.scorer.count} molecules have been scored")

    def run(self):
        self.write_status()
        for _ in range(1, self.iterations):
            self.train()
            df = self.predict()
            df = self.extract_best(df)
            df["scores"] = self.scorer.score(df.smi, self.cpu, self.gpu)
            df["generation"] = self.gen
            self.save_data(df)
            self.taken.update(df.unique)
            self.write_status()

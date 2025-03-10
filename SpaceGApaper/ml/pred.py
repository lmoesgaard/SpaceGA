import os
import torch
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from ml.train import get_model
from utils import split_array, smi2array

class MlScreen:
    def __init__(self, model_name, model_path, cpu, gpu, prefixes, population, basedir, maxmodels):
        self.model_name = model_name
        self.model_path = model_path
        self.gpu = gpu
        self.cpu = min(cpu, gpu * maxmodels)
        self.prefixes = prefixes
        self.population = population
        self.basedir = basedir

    def pred_parallel(self):
        data = Parallel(n_jobs=self.cpu)(delayed(self.run_pred)(i) for i in range(self.cpu))
        return data

    def get_model(self, i):
        model = get_model(self.model_name, 1024)
        model.load_state_dict(torch.load(self.model_path))
        model = model.to(torch.device(f"cuda:{i}"))
        return model

    def run_pred(self, i):
        gpu = i % self.gpu
        model = self.get_model(gpu)
        device = torch.device(f"cuda:{gpu}")
        prefixes = [(n, prefix) for n, prefix in enumerate(self.prefixes) if (n % self.cpu == i)]
        preds = None

        for n, prefix in prefixes:
            try:
                # Load and preprocess data
                file = os.path.join(self.basedir, f"fps/{prefix}.npz")
                a = np.load(file, allow_pickle=True)['array'].astype(np.float32)
                a = torch.tensor(a).to(device)

                # Model prediction
                with torch.no_grad():
                    pred = model(a).flatten().to("cpu").numpy()

                # Create DataFrame
                labels = [f"{prefix}_{i}" for i in range(len(pred))]
                df = pd.DataFrame({"unique": labels, "pred": pred})
                df["prefix"] = prefix

                # Update predictions
                preds = pd.concat([preds, df])
                preds = preds.sort_values("pred", ascending=False).head(self.population)

                # Free up memory
                del a, labels, df
                torch.cuda.empty_cache()
            except:
                print(f"Error processing prefix {prefix} on GPU {i}")

        del model
        torch.cuda.empty_cache()
        return preds

class PredictScores:
    def __init__(self, model_name, model_path, cpu, gpu):
        self.model_name = model_name
        self.model_path = model_path
        self.gpu = gpu
        self.cpu = cpu

    def get_model(self):
        model = get_model(self.model_name, 1024)
        model.load_state_dict(torch.load(self.model_path))
        model = model.to(torch.device(f"cuda:0"))
        return model

    def smis2a(self, smi_lst):
        a = np.array([smi2array(s) for s in smi_lst])
        return a

    def run_pred(self, smi_lst):
        model = self.get_model()
        device = torch.device(f"cuda:0")

        # Preprocess data
        smi_lst = split_array(smi_lst, self.cpu)
        a = Parallel(n_jobs=self.cpu)(delayed(self.smis2a)(lst) for lst in smi_lst)
        a = np.concatenate(a).astype(np.float32)
        a = torch.tensor(a).to(device)

        # Model prediction
        with torch.no_grad():
            pred = model(a).flatten().to("cpu").numpy()

        # Free up memory
        del a, model
        torch.cuda.empty_cache()
        return pred
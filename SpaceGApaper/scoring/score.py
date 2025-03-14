from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import math
import pandas as pd
import os
from rdkit import DataStructs
import shutil

from utils import split_df, smi2fp, logP, generate_random_name, submit_job
from scoring.autodock import AutodockGpu
from scoring.get_scores import process_files
from setup import read_config


def load_class(class_name, module_name):
    module = __import__(module_name, fromlist=[class_name])
    return getattr(module, class_name)


class Scorer(ABC):
    @abstractmethod
    def __init__(self, arguments):
        pass

    def score(self, smi_lst, cpu, gpu):
        pass


class FPSearch(Scorer):
    def __init__(self, arguments):
        self.count = 0
        self.ref = smi2fp(arguments["smiles"])

    def scorer(self, smi):
        fp = smi2fp(smi)
        return DataStructs.TanimotoSimilarity(self.ref, fp)

    def score(self, smi_lst, cpu, gpu):
        scores = Parallel(n_jobs=cpu)(delayed(self.scorer)(smi) for smi in smi_lst)
        self.count += len(smi_lst)
        return scores


class LogPSearch(Scorer):
    def __init__(self, arguments):
        self.count = 0

    @staticmethod
    def scorer(smi):
        return logP(smi)

    def score(self, smi_lst, cpu, gpu):
        scores = Parallel(n_jobs=cpu)(delayed(self.scorer)(smi) for smi in smi_lst)
        self.count += len(smi_lst)
        return scores


class DockSearch(Scorer):
    def __init__(self, arguments):
        self.fld_file = arguments["fld_file"]
        self.autodock = arguments["autodock"]
        self.workdir = arguments["workdir"]
        self.obabel = arguments["obabel"]
        self.count = 0
        self.scratch = None

    def score_files(self, pdbqt_files, i, gpu):
        autodocking = AutodockGpu(self.scratch, self.autodock, self.fld_file, i, pdbqt_files, gpu)
        return autodocking.run()

    def write_smi(self, df, smi_name):
        df[["smi", "name"]].to_csv(smi_name, header=None, index=False, sep=" ")

    def confgen(self, df, prefix):
        smi_name = os.path.join(self.scratch, f"{prefix}_mols.smi")
        pdbqt_base = os.path.join(self.scratch, f"{prefix}_mols.pdbqt")
        self.write_smi(df, smi_name)
        del df
        job = f"{self.obabel} -ismi {smi_name} -O {smi_name} -p"
        submit_job(job)
        job = f"{self.obabel} -ismi {smi_name} -O {pdbqt_base} -m --seed 1 --gen3d --fast"
        submit_job(job)
        del smi_name, pdbqt_base, prefix

    def find_files(self, suffix):
        return [os.path.join(self.scratch, file) for file in os.listdir(self.scratch) if file.endswith(suffix)]

    @staticmethod
    def process_output(outputs, df):
        best_scores = {name: -1 * score for output in outputs for name, score in output}
        scores = [best_scores.get(str(name), 0) for name in df["name"]]
        return scores

    def score_batch(self, df, cpu, gpu):
        self.scratch = os.path.join(self.workdir, generate_random_name(6))
        try:
            os.mkdir(self.scratch)
        except:
            pass
        # generate conformers
        Parallel(n_jobs=cpu, timeout=None)(
            delayed(self.confgen)(subset, i) for i, subset in enumerate(split_df(df, cpu)))
        pdbqt_files = self.find_files("pdbqt")
        # run docking
        Parallel(n_jobs=gpu, timeout=None)(delayed(self.score_files)(pdbqt_files, i, gpu) for i in range(gpu))
        xml_files = self.find_files("xml")
        # process output
        outputs = Parallel(n_jobs=cpu, timeout=None)(
            delayed(process_files)(xml_files, i, cpu) for i in range(cpu))
        scores = self.process_output(outputs, df)
        # clean up
        shutil.rmtree(self.scratch)
        return scores

    def score(self, smi_lst, cpu, gpu):
        df = pd.DataFrame(smi_lst).reset_index()
        df.columns = ["name", "smi"]
        n_batches = math.ceil(df.shape[0]/(gpu*200))
        scores = []
        for subset in split_df(df, n_batches):
            scores += self.score_batch(subset, cpu, gpu)
        self.count += len(smi_lst)
        return scores

class CustomSearch:
    def __init__(self, arguments):
        class_inputs = read_config(arguments["config"])
        class_inputs["workdir"] = arguments["workdir"]
        self.method = load_class(class_name=arguments["class_name"], module_name=arguments["module_name"])(class_inputs)
        self.count = 0

    def score(self, smi_lst, cpu, gpu):
        self.count += 1
        return self.method.score(smi_lst, cpu, gpu)
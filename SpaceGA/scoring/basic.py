from abc import ABC, abstractmethod
from joblib import Parallel, delayed
from rdkit import DataStructs

from utils import smi2fp, logP


class Scorer(ABC):
    @abstractmethod
    def __init__(self, arguments):
        pass

    def score(self, smi_lst, name_lst):
        pass


class LogPSearch(Scorer):
    def __init__(self, arguments):
        self.cpu = arguments["cpu"]

    @staticmethod
    def scorer(smi):
        return logP(smi)

    def score(self, smi_lst, name_lst):
        scores = Parallel(n_jobs=self.cpu)(delayed(self.scorer)(smi) for smi in smi_lst)
        return scores


class FPSearch(Scorer):
    def __init__(self, arguments):
        self.cpu = arguments["cpu"]
        self.ref = smi2fp(arguments["scoring_inputs"]["smiles"])

    def scorer(self, smi):
        fp = smi2fp(smi)
        return DataStructs.TanimotoSimilarity(self.ref, fp)

    def score(self, smi_lst, name_lst):
        scores = Parallel(n_jobs=self.cpu)(delayed(self.scorer)(smi) for smi in smi_lst)
        return scores
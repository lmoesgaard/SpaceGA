from abc import ABC, abstractmethod
import random


class Scorer(ABC):
    @abstractmethod
    def __init__(self, arguments):
        pass

    def score(self, smi_lst, cpu, gpu):
        pass

class RandomSearch(Scorer):
    def __init__(self, arguments):
        self.count = 0

    def score(self, smi_lst, cpu, gpu):
        scores = [random.random() for _ in smi_lst]
        self.count += len(smi_lst)
        return scores
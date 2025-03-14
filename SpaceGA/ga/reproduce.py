from rdkit import Chem, rdBase
import random
import sys
from joblib import Parallel, delayed, load
from rdkit import Chem, rdBase
import random
import pandas as pd

from SpaceGA.utils import smifile2df, mol2array
from SpaceGA.ga import crossover, neutralize_molecules
from SpaceGA.sl.spacelight import Simsearch

class Reproduce:
    def pick_random_item(self, n, col):
        weights = self.population["scores"] - min(self.population["scores"])
        sel = self.population.sample(n, weights=weights)
        return list(sel[col])
    
    def generate_molecule(self, n_tries=10):
        rdBase.DisableLog("rdApp.error")
        do_cross = random.random() < self.crossover_rate

        while n_tries > 0:
            n_tries -= 1

            # Select parent molecules and apply crossover if needed
            parents = self.pick_random_item(2 if do_cross else 1, "mol")
            mol = crossover.crossover(*parents) if do_cross else parents[0]

            if mol is None:
                continue

            mol = neutralize_molecules([mol])[0]
            if mol is None or not self.search.filter.filter_mol(mol):
                continue

            smi = Chem.MolToSmiles(mol)
            children = self.search.search(smi, scramble=not do_cross)

            if children is not None:
                return children

        return None

    def reproduce(self):
        self.taken = set(self.taken)
        if self.gen == 1:
            offspring = smifile2df(self.i).sample(self.generation_size)
            offspring["mol"] = offspring.apply(lambda x: Chem.MolFromSmiles(x.smi), axis=1)
            if self.al:
                offspring["fp"] = offspring.mol.apply(mol2array)
        else:
            self.search = Simsearch(self.__dict__, taken=self.taken)
            offspring = Parallel(n_jobs=self.sl_cpu)(delayed(self.generate_molecule)() for _ in range(self.p_size))
            if not offspring or all(x is None for x in offspring):
                print("Failed to reproduce: No valid offspring generated.")
                sys.exit()
            offspring = pd.concat(offspring)
            offspring = offspring.drop_duplicates("name")
            # Filter offspring further if too many molecules have been found
            if offspring.shape[0] > self.generation_size:
                if self.al:
                    offspring["fp"] = offspring.mol.apply(lambda x: mol2array(x))
                    model = load(self.model_path.format(self.gen-1))
                    offspring["pred"] = model.predict(offspring.fp.to_list())
                    offspring = offspring.nlargest(self.generation_size, "pred")
                else:
                    offspring = offspring.sample(self.generation_size)
        self.taken.update(offspring["name"])
        self.taken = list(self.taken)
        offspring["generation"] = self.gen
        self.search = None
        return offspring
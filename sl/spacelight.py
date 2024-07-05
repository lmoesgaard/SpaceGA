import os
import pandas as pd
from rdkit import Chem
from utils import submit_job_info, generate_random_name


class simsearch():
    def __init__(self, scratch, space, propfilter, children, spacelight):
        self.scratch = scratch
        self.space = space
        self.filter = propfilter
        self.children = children
        self.sample = int(children*100)
        self.taken = set()
        self.spacelight = spacelight
        self.allowed = {"C", "O", "N", "I", "Br", "Cl", "F", "S"}

    def check_weird_elements(self, m):
        atoms = {a.GetSymbol() for a in m.GetAtoms()}
        return len(atoms.difference(self.allowed)) > 0

    def search(self, smi, drop_first=True, scrample=True):
        # run spacelight
        dst = os.path.join(self.scratch, f"{generate_random_name(6)}.csv")
        job = f'{self.spacelight} --thread-count 1 --max-nof-results {self.sample} -i "{smi}" -s {self.space} -O {dst} -f ECFP4'
        submit_job_info(job)
        try:
            # read output
            df = pd.read_csv(dst)
            if drop_first:
                # drop first row - likely the input molecule
                df = df.drop(0)
            # remove molecules that have already been scored
            df = df[df.apply(lambda x: x["result-name"] not in self.taken, axis=1)]
            # filter off molecules that don't have desired properties
            df["mol"] = df.apply(lambda x: Chem.MolFromSmiles(x["#result-smiles"]), axis=1)
            df = df[self.filter.filter_mol_lst(df["mol"])]
            df = df[[not self.check_weird_elements(m) for m in df.mol]]
            # pick a set of children - the higher similarity, the higher the probability
            if df.shape[0] > self.children:
                if scrample:
                    df = df.sample(self.children, weights="similarity")
                else:
                    df = df.head(self.children)
            # remove output file
            os.remove(dst)
            # return the children
            df = df[["#result-smiles", "result-name", "mol"]]
            df.columns = ["smi", "name", "mol"]
            return df
        except:
            return None
import os
import pandas as pd
from utils import submit_job_info, generate_random_name


class simsearch():
    def __init__(self, scratch, space, propfilter, children, f_comp, spacelight, al=False):
        self.scratch = scratch
        self.space = space
        self.filter = propfilter
        self.children = children
        self.sample = int(children*f_comp)
        self.taken = set()
        self.spacelight = spacelight
        self.al = al

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
        except:
            print("No output from SpaceLight")
            return None
        # remove molecules that have already been scored
        df = df[df.apply(lambda x: x["result-name"] not in self.taken, axis=1)]
        # remove molecules that have already been scored
        df = df.drop_duplicates("result-name")
        # filter off molecules that don't have desired properties
        mask, mols, smis = self.filter.substructure_filter(df["#result-smiles"].to_list())
        df = df[mask].copy()
        df["mol"], df["smi"] = (mols, smis)
        if df.shape[0] == 0:
            print("All molecules had bad substructures")
            return None
        df = df[self.filter.filter_mol_lst(df["mol"])]
        if df.shape[0] == 0:
            print("No molecules had desired properites")
            return None
        # pick a set of children - the higher similarity, the higher the probability
        if df.shape[0] > self.children and not self.al:
            if scrample:
                df = df.sample(self.children, weights="fingerprint-similarity")
            else:
                df = df.head(self.children)
        # remove output file
        os.remove(dst)
        # return the children
        df = df[["smi", "result-name", "mol"]]
        df.columns = ["smi", "name", "mol"]
        return df
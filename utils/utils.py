import subprocess
import importlib
import string
import random
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, rdFingerprintGenerator


def get_scorer(mode, arguments):
    module = importlib.import_module("scoring")
    class_ = getattr(module, mode)
    scorer = class_(arguments)
    return scorer


def submit_job_info(job):
    subprocess.run(job, shell=True)


def submit_job(job):
    subprocess.run(job, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def generate_random_name(length=6):
    characters = string.ascii_lowercase + string.digits
    random_name = ''.join(random.choice(characters) for _ in range(length))
    return random_name


def split_array(a, n):
    chunk_size = len(a) // n
    remainder = len(a) % n
    return [a[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(n)]


def split_df(df, n):
    chunk_size = len(df) // n
    remainder = len(df) % n
    return [df.iloc[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(n)]


def smi2fp(smi):
    m = Chem.MolFromSmiles(smi)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(mol)
    return fp


def mol2fp(m):
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(mol)
    return fp


def logP(smi):
    mol = Chem.MolFromSmiles(smi)
    return Crippen.MolLogP(mol)


def smifile2df(smifile):
    df = pd.read_csv(smifile, header=None, index_col=None, names=["smi", "name"], sep=" ")
    return df


def sim_filter(mol_lst, pop_size, cutoff=0.35):
    fps = [mol2fp(mol_lst[0])]
    mask = np.zeros(len(mol_lst), dtype=bool)
    mask[0] = True
    ind = 1
    while mask.sum() < pop_size and ind < len(mol_lst):
        mol = mol_lst[ind]
        fp = mol2fp(mol)
        sim = np.max(DataStructs.BulkTanimotoSimilarity(fp, fps))
        if sim < cutoff:
            fps.append(fp)
            mask[ind] = True
        ind += 1
    return mask


def smi2array(smi):
    mol = Chem.MolFromSmiles(smi)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048).GetFingerprint(mol)
    array = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, array)
    return array


def get_train_mask(dataset, split_ratio):
    total_samples = len(dataset)  # need the length of dataset
    num_train_samples = int(split_ratio[0] * total_samples)  # find the number of samples to put in trainset
    num_val_samples = int(split_ratio[1] * total_samples)
    num_test_samples = total_samples - num_train_samples - num_val_samples
    splits = np.array([0] * num_train_samples + [1] * num_val_samples + [2] * num_test_samples)
    np.random.shuffle(splits)  # shuffle train and test values
    return {"train": splits == 0, "val": splits == 1, "test": splits == 2}

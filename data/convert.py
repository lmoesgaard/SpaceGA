from joblib import Parallel, delayed
import pandas as pd
from rdkit import Chem, DataStructs, rdFingerprintGenerator
import numpy as np
import os
import shutil
import sys
import argparse


class Converter:
    def __init__(self, smi, fps_path, smis_path, num_cpus, num_batches):
        self.smi = smi
        self.fps_path = fps_path
        self.smis_path = smis_path
        self.num_cpus = num_cpus
        self.num_batches = num_batches


    def make_batch_file(self, i):
        df = {"smi": [], "name": []}
        fps = []
        with open(self.smi) as f:
            for j, line in enumerate(f):
                if j % self.num_batches == i:
                    data = line.strip().split(" ")
                    if len(data) == 2:
                        smi, name = data
                        try:
                            fp, smi = process(smi)
                        except:
                            print(f"Failed with SMILES: {smi}")
                            continue
                        df["smi"].append(smi)
                        df["name"].append(name)
                        fps.append(fp)
                    else:
                        print(f"Line did not have two columns: {line}")
        df = pd.DataFrame(df)
        df.to_parquet(os.path.join(self.smis_path, f"{i}.parquet"))
        fps = np.array(fps)
        np.savez_compressed(os.path.join(self.fps_path, f"{i}.npz"), array=fps)


def create_directories(output_path, allow_overwrite):
    fps_path = os.path.join(output_path, 'fps')
    smis_path = os.path.join(output_path, 'smis')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        os.makedirs(fps_path)
        os.makedirs(smis_path)
        print(f"Directories created: {fps_path}, {smis_path}")
    else:
        if allow_overwrite:
            if os.path.exists(fps_path):
                shutil.rmtree(fps_path)
            if os.path.exists(smis_path):
                shutil.rmtree(smis_path)
            os.makedirs(fps_path)
            os.makedirs(smis_path)
            print(f"Directories recreated: {fps_path}, {smis_path}")
        else:
            try:
                os.makedirs(fps_path)
                os.makedirs(smis_path)
            except:
                print(f"Directories existed\nQuitting")
                exit()
    return fps_path, smis_path

def process(smi):
    mol = Chem.MolFromSmiles(smi)  # Convert SMILES to RDKit molecule
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024).GetFingerprint(mol)  # Generate Morgan fingerprint
    array = np.zeros((0, ), dtype=np.int8)  # Initialize an empty Numpy array
    DataStructs.ConvertToNumpyArray(fp, array)  # Convert fingerprint to Numpy array
    new_smi = Chem.MolToSmiles(mol)
    return array, new_smi

def main():
    parser = argparse.ArgumentParser(description='Process SMILES file' epilog='Example: python convert.py -O -i ZINC.smi -n 5 --cpu 5')
    parser.add_argument('-O', action='store_true', help='Allow overwriting of directories')
    parser.add_argument('-i', type=str, required=True, help='Input space-separated SMILES file without header')
    parser.add_argument('-o', type=str, default=".", help='Output destination to create directories fps and smis')
    parser.add_argument('--cpu', type=int, default=1, help='Number of processes the script is allowed to use')
    parser.add_argument('-n', type=int, default=1, help='Number of batches to split data into')

    args = parser.parse_args()

    if not os.path.isfile(args.i):
        print(f"Error: Input file {args.i} does not exist.")
        sys.exit(1)

    fps_path, smis_path = create_directories(args.o, args.O)

    num_cpus = args.cpu
    num_batches = args.n
    print(f"Number of processes allowed: {num_cpus}")
    print(f"Number of batches: {num_batches}")

    c = Converter(smi=args.i, fps_path=fps_path, smis_path=smis_path, num_cpus=num_cpus, num_batches=num_batches)
    Parallel(n_jobs=num_cpus)(delayed(c.make_batch_file)(i) for i in range(num_batches))

if __name__ == '__main__':
    main()
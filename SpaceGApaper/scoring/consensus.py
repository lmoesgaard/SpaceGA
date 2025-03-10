from abc import ABC, abstractmethod
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import os
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
import shutil

from utils import split_df, generate_random_name


def read_sdf(file):
    df = []
    with open(file) as f:
        current = {"pose": ""}
        info = False
        for line in f:
            if len(current["pose"]) == 0:
                current["name"] = line.strip()
            current["pose"] += line
            if line.startswith("> <"):
                name = line[3:-2]
                info = True
            elif info:
                try:
                    current[name] = float(line.strip())
                except:
                    current[name] = line.strip()
                info = False
            elif line.startswith("$$$"):
                df.append(current)
                current = {"pose": ""}
    return pd.DataFrame(df)


def run_smina(arguments, input_sdf):
    output_sdf = input_sdf.split(".")[0] + "_out.sdf"
    job = f'{arguments["smina"]} -r {arguments["receptor"]} -l {input_sdf} -o {output_sdf} '
    job += f'--center_x {arguments["center"][0]} --center_y {arguments["center"][1]} --center_z {arguments["center"][2]} '
    job += f'--size_x {arguments["size"][0]} --size_y {arguments["size"][1]} --size_z {arguments["size"][2]} '
    job += "--cpu 1 --num_modes 1 --exhaustiveness 1"
    job += " > /dev/null 2>&1"
    os.system(job)


def calc_rmsd(df):
    RDLogger.DisableLog('rdApp.warning')
    rmsd = []
    for i, row in df.iterrows():
        mol1 = Chem.MolFromMolBlock(row.pose_x)
        mol2 = Chem.MolFromMolBlock(row.pose_y)
        mol1_coords = mol1.GetConformer().GetPositions()
        mol2_coords = mol2.GetConformer().GetPositions()
        mcs = rdFMCS.FindMCS([mol1, mol2], completeRingsOnly=True)
        sub_mol = Chem.MolFromSmarts(mcs.smartsString)
        RMSDs = []
        for mol1_ids in mol1.GetSubstructMatches(sub_mol, useChirality=False, uniquify=False):
            for mol2_ids in mol2.GetSubstructMatches(sub_mol, useChirality=False, uniquify=False):
                mol1_coordinates = np.array([mol1_coords[i] for i in mol1_ids])
                mol2_coordinates = np.array([mol2_coords[i] for i in mol2_ids])
                RMSD = np.linalg.norm(mol1_coordinates - mol2_coordinates, axis=1).mean()
                RMSDs.append(RMSD)
        rmsd.append(min(RMSDs))
    df["rmsd"] = rmsd
    return df


class Scorer(ABC):
    @abstractmethod
    def __init__(self, arguments):
        pass

    def score(self, smi_lst, cpu, gpu):
        pass


class CustomScorer(Scorer):
    def __init__(self, arguments):
        self.workdir = arguments["workdir"]
        self.grid = arguments["grid"]
        self.schrodinger = arguments["schrodinger"]
        self.pose_dir = os.path.join(arguments["workdir"], "poses")
        self.arguments = arguments

        try:
            os.mkdir(self.pose_dir)
        except:
            pass
        self.iteration = 1
        self.glide_distribution = None
        self.smina_distribution = None
        self.count = 0
        self.scratch = None

    def write_smi(self, df, smi_name):
        df[["smi", "names"]].to_csv(smi_name, header=None, index=False, sep=" ")

    def run_ligprep(self, smi_lst, cpu):
        df = pd.DataFrame(smi_lst, columns=["smi"])
        df["names"] = np.arange(df.shape[0])
        smi_file = os.path.join(self.scratch, "ligs.smi")
        self.write_smi(df, smi_file)
        input_file = os.path.join(self.scratch, "ligprep.inp")
        with open(input_file, "w") as f:
            f.write(f"INPUT_FILE_NAME {smi_file}\n")
            f.write("MAX_ATOMS   500\n")
            f.write("FORCE_FIELD   14\n")
            f.write("PH_THRESHOLD   1.0\n")
            f.write("EPIK   yes\n")
            f.write("EPIKX   no\n")
            f.write("EPIK_METAL_BINDING   no\n")
            f.write("INCLUDE_ORIGINAL_STATE   no\n")
            f.write("DETERMINE_CHIRALITIES   no\n")
            f.write("IGNORE_CHIRALITIES   no\n")
            f.write("NUM_STEREOISOMERS   1\n")
            f.write(f"OUT_SD   ligs.sdf\n")
        cmd = f"{os.path.join(self.schrodinger, 'ligprep')} -inp ligprep.inp -JOBNAME ligprep -HOST localhost:{cpu} -WAIT\n"
        return cmd

    def run_glide(self, cpu):
        glide_file = os.path.join(self.scratch, "glide.in")
        with open(glide_file, "w") as f:
            f.write("FORCEFIELD   OPLS_2005\n")
            f.write(f"GRIDFILE   {self.grid}\n")
            f.write("LIGANDFILE   ligs.sdf\n")
            f.write("NREPORT   1\n")
            f.write("POSE_OUTTYPE   ligandlib_sd\n")
            f.write("PRECISION   SP\n")
            f.write("POSTDOCKSTRAIN   True\n")
        cmd = f"{os.path.join(self.schrodinger, 'glide')} {glide_file} -OVERWRITE -adjust -HOST localhost:{cpu} -TMPLAUNCHDIR -WAIT\n"
        cmd += f"mv glide_subjob_poses.zip {os.path.join(self.pose_dir, str(self.iteration) + '_poses.zip')}\n"
        cmd += f"unzip {os.path.join(self.pose_dir, str(self.iteration) + '_poses.zip')} -d {self.pose_dir} \n"
        cmd += f"cd {self.pose_dir} \n"
        cmd += 'for file in *_raw.sdfgz; do\n\tmv "$file" "${file/_raw.sdfgz/.sdf.gz}"\ndone\n'
        cmd += "gunzip *.sdf.gz\n"
        cmd += f"cat glide*sdf > {self.iteration}_glide_poses.sdf\n"
        cmd += "rm *zip\n rm glide*sdf\n"
        glide_poses = os.path.join(self.pose_dir, f"{self.iteration}_glide_poses.sdf")
        return cmd, glide_poses

    def run_smina_parallel(self, cpu):
        df = read_sdf(os.path.join(self.scratch, "ligs.sdf"))
        sdf_files = []
        for i, subset in enumerate(split_df(df, cpu)):
            lig_file = os.path.join(self.scratch, f"{i}_ligs.sdf")
            with open(lig_file, "w") as f:
                for pose in subset.pose:
                    f.write(pose)
            sdf_files.append(lig_file)
        Parallel(n_jobs=cpu, timeout=None)(delayed(run_smina)(self.arguments, lig_in) for lig_in in sdf_files)
        cmd = f"cat {self.scratch}/*out.sdf > {self.pose_dir}/{self.iteration}_smina_poses.sdf"
        os.system(cmd)
        return os.path.join(self.pose_dir, f"{self.iteration}_smina_poses.sdf")

    def run_cmd(self, cmd):
        bash_file = os.path.join(self.scratch, "job.sh")
        with open(bash_file, "w") as f:
            f.write(cmd)
        os.system(f"bash {bash_file}")

    def process_results(self, smina_poses, glide_poses, cpu):
        smina_df = read_sdf(smina_poses).sort_values("minimizedAffinity").drop_duplicates("name")
        glide_df = read_sdf(glide_poses).sort_values("r_i_docking_score").drop_duplicates("name")
        df = pd.merge(smina_df, glide_df, on="name")
        df = Parallel(n_jobs=cpu, timeout=None)(delayed(calc_rmsd)(subset) for subset in split_df(df, cpu))
        df = pd.concat(df)
        if self.glide_distribution is None:
            self.glide_distribution = {"mean": df.r_i_docking_score.mean(),
                                       "std": df.r_i_docking_score.std()}
        if self.smina_distribution is None:
            self.smina_distribution = {"mean": df.minimizedAffinity.mean(),
                                       "std": df.minimizedAffinity.std()}
        df["score"] = -0.5*((df.minimizedAffinity-self.smina_distribution["mean"])/self.smina_distribution["std"] +
                            (df.r_i_docking_score-self.glide_distribution["mean"])/self.glide_distribution["std"])
        df.score += -1*(df.rmsd < 2).astype(int)
        return df

    def score(self, smi_lst, cpu, gpu):
        self.scratch = os.path.join(self.workdir, generate_random_name(6))
        os.mkdir(self.scratch)
        cmd = f"cd {self.scratch}\n\n"
        cmd += self.run_ligprep(smi_lst, cpu)
        glide_cmd, glide_poses = self.run_glide(cpu)
        cmd += glide_cmd
        self.run_cmd(cmd)
        smina_poses = self.run_smina_parallel(cpu)
        df = self.process_results(smina_poses, glide_poses, cpu)
        scores = [-3]*len(smi_lst)
        for _, row in df.iterrows():
            scores[int(row["name"])] = row.score
        self.count += len(smi_lst)
        self.iteration += 1
        shutil.rmtree(self.scratch)
        return scores

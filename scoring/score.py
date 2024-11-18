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

class glide(Scorer):
    def __init__(self, arguments):
        self.workdir = arguments["workdir"]
        self.grid = arguments["grid"]
        self.schrodinger = arguments["schrodinger"]
        self.pose_dir = os.path.join(arguments["workdir"], "poses")
        try:
            os.mkdir(self.pose_dir)
        except:
            pass
        self.iteration = 1
        self.count = 0
        self.scratch = None

    def write_smi(self, df, smi_name):
        df[["smi", "name"]].to_csv(smi_name, header=None, index=False, sep=" ")

    def run_ligprep(self, smi_lst, cpu):
        df = pd.DataFrame(smi_lst).reset_index()
        df.columns = ["name", "smi"]
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
            f.write(f"OUT_MAE   ligs.mae\n")
        cmd = f"{os.path.join(self.schrodinger, 'ligprep')} -inp ligprep.inp -JOBNAME ligprep -HOST localhost:{cpu} -WAIT\n"
        return cmd      

    def run_glide(self, cpu):
        glide_file = os.path.join(self.scratch, "glide.in")
        with open(glide_file, "w") as f:
            f.write("FORCEFIELD   OPLS_2005\n")
            f.write(f"GRIDFILE   {self.grid}\n")
            f.write("LIGANDFILE   ligs.mae\n")
            f.write("NREPORT   1\n")
            f.write("POSE_OUTTYPE   ligandlib_sd\n")
            f.write("PRECISION   SP\n")
        cmd = f"{os.path.join(self.schrodinger, 'glide')} {glide_file} -OVERWRITE -adjust -HOST localhost:{cpu} -TMPLAUNCHDIR -WAIT\n"
        cmd += f"mv glide_subjob_poses.zip {os.path.join(self.pose_dir, str(self.iteration) + '_poses.zip')}\n"
        return cmd      

    def process_glide_results(self, csv_file, smi):
        results = {i: 0 for i in range(len(smi))}
        df = pd.read_csv(csv_file)
        df = df[["title", "r_i_docking_score"]]
        for _, row in df.iterrows():
            results[int(row["title"])] = min(results[int(row["title"])], row["r_i_docking_score"])
        return [-1*results[i] for i in range(len(smi))]

    def run_cmd(self, cmd):
        bash_file = os.path.join(self.scratch, "job.sh")
        with open(bash_file, "w") as f:
            f.write(cmd)
        os.system(f"bash {bash_file}")

    def score(self, smi_lst, cpu, gpu):
        self.scratch = os.path.join(self.workdir, generate_random_name(6))
        os.mkdir(self.scratch)
        cmd = f"cd {self.scratch}\n\n"
        cmd += self.run_ligprep(smi_lst, cpu)
        cmd += self.run_glide(cpu)
        self.run_cmd(cmd)
        csv_file = os.path.join(self.scratch, "glide.csv")
        scores = self.process_glide_results(csv_file, smi_lst)
        self.count += len(smi_lst)
        self.iteration += 1
        shutil.rmtree(self.scratch)
        return scores

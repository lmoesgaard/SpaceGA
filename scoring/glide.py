from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import os
import shutil

from setup import read_config
from utils import generate_random_name


class Scorer(ABC):
    @abstractmethod
    def __init__(self, arguments):
        pass

    def score(self, smi_lst, cpu, gpu):
        pass


class GlideScorer(Scorer):
    """
    This class requires a dictionary as input with the following keys:

    - `workdir` (str): Path to the working directory. Typically, this should be
      the same as the parent directory when running a prioritization algorithm.
    - `grid` (str): Path to the Glide grid file (e.g., /path/to/glide_grid.zip).
    - `schrodinger` (str): Path to the Schrodinger installation directory.

    Example input dictionary:
    {
        "workdir": "/path/to/workdir",
        "grid": "/path/to/glide_grid.zip",
        "schrodinger": "/path/to/schrodinger/installation"
    }
    """
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
        cmd = f"{os.path.join(self.schrodinger, 'glide')} {glide_file}"
        cmd += f" -OVERWRITE -adjust -HOST localhost:{cpu} -TMPLAUNCHDIR -WAIT\n"
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


def load_class(arguments):
    module = __import__(arguments["module_name"], fromlist=[arguments["class_name"]])
    return getattr(module, arguments["class_name"])


class CustomSearch:
    def __init__(self, arguments):
        arguments = read_config(arguments["config"])
        self.method = load_class(arguments)()
        self.count = 0

    def score(self, smi_lst, cpu, gpu):
        self.count += 1
        return self.method.score(smi_lst, cpu, gpu)

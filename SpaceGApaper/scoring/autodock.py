import os

from utils.utils import submit_job


class AutodockGpu():
    def __init__(self, scratch, autodock_path, fld_file, i, pdbqt_files, gpu):
        self.autodock_path = autodock_path
        self.fld_file = fld_file
        self.i = i
        self.pdbqt_files = pdbqt_files
        self.scratch = scratch
        self.gpu = gpu

    def name_file(self, name):
        return os.path.join(self.scratch, name)

    def write_batch_file(self):
        batch_file = self.name_file(f"ligand_batch_{self.i}.txt")
        with open(batch_file, "w") as f:
            f.write(f"{self.fld_file}\n")
            for n, file in enumerate(self.pdbqt_files):
                if n % self.gpu == self.i:
                    f.write(f"{file}\n")
                    f.write(f"{file.split('.')[0].split('/')[-1]}\n")
        return batch_file

    def run_docking(self):
        batch_file = self.write_batch_file()
        out_base = self.name_file(f"out_{self.i}")
        job = f"{self.autodock_path} --filelist {batch_file} -seed 1,1,1 --nrun 10 --nev 1000000 --resnam {out_base} --devnum {self.i+1}"
        submit_job(job)

    def run(self):
        self.run_docking()

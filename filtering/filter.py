import subprocess, os
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from utils import generate_random_name

phys_filters = {"logP": Descriptors.MolLogP,
                "Mw": Descriptors.MolWt,
                "HBA": Descriptors.NumHAcceptors,
                "HBD": Descriptors.NumHDonors,
                "Rings": Descriptors.RingCount,
                "RotB": Descriptors.NumRotatableBonds,
                }


def smi_to_neutral_mol(smi):
    mol = Chem.MolFromSmiles(smi)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    if len(at_matches_list) > 0:
        for at_idx in at_matches_list:
            atom = mol.GetAtomWithIdx(at_idx)
            chg = atom.GetFormalCharge()
            hcount = atom.GetTotalNumHs()
            atom.SetFormalCharge(0)
            atom.SetNumExplicitHs(hcount - chg)
            atom.UpdatePropertyCache()
    return mol


class IntervalFilter:
    def __init__(self, function):
        self.lims = {"min": -float('inf'), "max": float('inf')}
        self.function = function

    def filter(self, mol):
        prop = self.function(mol)
        return self.lims["min"] <= prop <= self.lims["max"]


class Filtering:
    def __init__(self, arguments):
        self.allowed = {"C", "O", "N", "I", "Br", "Cl", "F", "S"}
        self.phys_filters = {}
        self.Lilly = False
        for argument in arguments:
            if argument[:3] in {"min", "max"}:
                lim = argument[:3]
                name = argument[3:]
                if name not in self.phys_filters:
                    self.phys_filters[name] = IntervalFilter(phys_filters[name])
                self.phys_filters[name].lims[lim] = arguments[argument]
        if "Lilly" in arguments:
            self.Lilly = arguments["Lilly"]
        params = FilterCatalogParams()
        if "PAINS" in arguments:
            if arguments["PAINS"]:
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS_A)
        if "BRENK" in arguments:
            if arguments["BRENK"]:
                params.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
        self.catalog = FilterCatalog(params)

    def check_weird_elements(self, m):
        atoms = {a.GetSymbol() for a in m.GetAtoms()}
        return len(atoms.difference(self.allowed)) > 0

    def run_medchem_rules(self, fname):
        cmd = f'{self.Lilly} -noapdm {fname}'
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        for line in iter(p.stdout.readline, b''):
            yield line.decode('utf-8')

    def run_lilly(self, smi_lst):
        fname = os.path.join(os.path.dirname(self.Lilly).split()[-1], f"{generate_random_name(6)}.smi")
        with open(fname, "w") as out:
            for i, smi in enumerate(smi_lst):
                out.write(f"{smi} i{i}\n")
        output = [int(line.rstrip().split(" ")[1][1:]) for line in self.run_medchem_rules(fname)]
        os.remove(fname)
        return output

    def filter_mol(self, mol):
        if self.check_weird_elements(mol):
            return False
        for filter in self.phys_filters:
            if not self.phys_filters[filter].filter(mol):
                return False
        return True

    def filter_mol_lst(self, mol_lst):
        mask = []
        for mol in mol_lst:
            mask.append(self.filter_mol(mol))
        return mask

    def filter_smi_lst(self, smi_lst):
        mask = []
        for smi in smi_lst:
            mol = Chem.MolFromSmiles(smi)
            mask.append(self.filter_mol(mol))
        return mask

    def filter_smi(self, smi):
        mol = Chem.MolFromSmiles(smi)
        return self.filter_mol(mol)

    def substructure_filter(self, smi_lst):
        mask = np.zeros(len(smi_lst)).astype(bool)
        good_idx = np.arange(len(smi_lst))
        if self.Lilly:
            good_idx = self.run_lilly(smi_lst)
        mask[good_idx] = True
        mols = []
        smis = []
        for idx in good_idx:
            mol = smi_to_neutral_mol(smi_lst[idx])
            if not self.catalog.HasMatch(mol):
                mols.append(mol)
                smis.append(Chem.MolToSmiles(mol))
            else:
                mask[idx] = False
        return mask, mols, smis

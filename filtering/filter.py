from rdkit import Chem
from rdkit.Chem import Descriptors

default_inputs = {"minlogP": -float('inf'),
                  "maxlogP": float('inf'),
                  "minMw": 0,
                  "maxMw": float('inf'),
                  "minHBA": 0,
                  "maxHBA": float('inf'),
                  "minHBD": 0,
                  "maxHBD": float('inf'),
                  "minRings": 0,
                  "maxRings": float('inf'),
                  "minRotB": 0,
                  "maxRotB": float('inf')
                  }


def apply_filter(mol, filter_type):
    prop = filter_type[2](mol)
    return filter_type[0] <= prop <= filter_type[1]


class Filtering:
    def __init__(self, arguments):
        filters = {**default_inputs, **arguments}
        self.logP_filter = [filters["minlogP"], filters["maxlogP"], Descriptors.MolLogP]
        self.Mw_filter = [filters["minMw"], filters["maxMw"], Descriptors.MolWt]
        self.HBA_filter = [filters["minHBA"], filters["maxHBA"], Descriptors.NumHAcceptors]
        self.HBD_filter = [filters["minHBD"], filters["maxHBD"], Descriptors.NumHDonors]
        self.ring_filter = [filters["minRings"], filters["maxRings"], Descriptors.RingCount]
        self.RotB_filter = [filters["minRotB"], filters["maxRotB"], Descriptors.NumRotatableBonds]

    def filter_mol(self, mol):
        if apply_filter(mol, self.logP_filter):
            if apply_filter(mol, self.Mw_filter):
                if apply_filter(mol, self.HBA_filter):
                    if apply_filter(mol, self.HBD_filter):
                        if apply_filter(mol, self.ring_filter):
                            if apply_filter(mol, self.RotB_filter):
                                return True
        return False

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

from typing import Union
import pkg_resources
import pandas as pd
from rdkit import Chem
from SpaceGA.filtering.sascorer import calculateScore

file_path = pkg_resources.resource_filename('SpaceGA', 'filtering/allert_collection.csv')
molecule_filters = [Chem.MolFromSmarts(smart) for smart in
                    pd.read_csv(file_path)["smarts"]]


def ring_ok(mol: Chem.Mol) -> bool:
    """ Checks that any rings in a molecule are OK

        :param mol: the molecule to check for rings
    """
    if not mol.HasSubstructMatch(Chem.MolFromSmarts('[R]')):
        return True

    ring_allene = mol.HasSubstructMatch(Chem.MolFromSmarts('[R]=[R]=[R]'))

    cycle_list = mol.GetRingInfo().AtomRings()
    max_cycle_length = max([len(j) for j in cycle_list])
    macro_cycle = max_cycle_length > 6

    double_bond_in_small_ring = mol.HasSubstructMatch(Chem.MolFromSmarts('[r3,r4]=[r3,r4]'))

    return not ring_allene and not macro_cycle and not double_bond_in_small_ring


def mol_ok(mol: Union[None, Chem.Mol]) -> bool:
    """ Returns of molecule on input is OK according to various criteria

      Criteria currently tested are:
        * check if RDKit can understand the smiles string
        * check if the size is OK
        * check if the molecule is sane

      :param mol: RDKit molecule
      :param molecule_options: the name of the filter to use
    """
    # break early of molecule is invalid
    if mol is None:
        return False

    # check for sanity
    try:
        Chem.SanitizeMol(mol)
    except (Chem.rdchem.AtomValenceException,
            Chem.rdchem.KekulizeException):
        return False

    test_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if test_mol is None:
        return False

    return 8 < mol.GetNumAtoms()


def synthesizeable(mol: Union[None, Chem.Mol], sa_cutoff=4.5) -> bool:
    for pattern in molecule_filters:
        if mol.HasSubstructMatch(pattern):
            return False
    sa = calculateScore(mol)
    if sa < sa_cutoff:
        return True

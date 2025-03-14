from typing import List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.warning')

from ga.molecule import mol_ok, ring_ok


def delete_atom() -> str:
    """ Returns a SMARTS string to delete an atom in a molecule """
    delete_smarts = ['[*:1]~[D1]>>[*:1]',
                     '[*:1]~[D2]~[*:2]>>[*:1]-[*:2]',
                     '[*:1]~[D3](~[*;!H0:2])~[*:3]>>[*:1]-[*:2]-[*:3]',
                     '[*:1]~[D4](~[*;!H0:2])(~[*;!H0:3])~[*:4]>>[*:1]-[*:2]-[*:3]-[*:4]',
                     '[*:1]~[D4](~[*;!H0;!H1:2])(~[*:3])~[*:4]>>[*:1]-[*:2](-[*:3])-[*:4]']
    p = [0.25, 0.25, 0.25, 0.1875, 0.0625]

    return np.random.choice(delete_smarts, p=p)


def append_atom() -> str:
    """ Returns a SMARTS string to append an atom to the molecule """
    choices = [['single', ['C', 'N', 'O', 'F', 'S', 'Cl', 'Br'], 7 * [1.0 / 7.0]],
               ['double', ['C', 'N', 'O'], 3 * [1.0 / 3.0]],
               ['triple', ['C', 'N'], 2 * [1.0 / 2.0]]]
    bond_order_probabilities = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=bond_order_probabilities)

    bond_order, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if bond_order == 'single':
        rxn_smarts = '[*;!H0:1]>>[*:1]X'.replace('X', '-' + new_atom)
    elif bond_order == 'double':
        rxn_smarts = '[*;!H0;!H1:1]>>[*:1]X'.replace('X', '=' + new_atom)
    else:
        rxn_smarts = '[*;H3:1]>>[*:1]X'.replace('X', '#' + new_atom)

    return rxn_smarts


def insert_atom() -> str:
    """ Returns a SMARTS string to insert an atom in a molecule """
    choices = [['single', ['C', 'N', 'O', 'S'], 4 * [1.0 / 4.0]],
               ['double', ['C', 'N'], 2 * [1.0 / 2.0]],
               ['triple', ['C'], [1.0]]]
    bond_order_probabilities = [0.60, 0.35, 0.05]

    index = np.random.choice(list(range(3)), p=bond_order_probabilities)

    bond_order, atom_list, p = choices[index]
    new_atom = np.random.choice(atom_list, p=p)

    if bond_order == 'single':
        rxn_smarts = '[*:1]~[*:2]>>[*:1]X[*:2]'.replace('X', new_atom)
    elif bond_order == 'double':
        rxn_smarts = '[*;!H0:1]~[*:2]>>[*:1]=X-[*:2]'.replace('X', new_atom)
    else:
        rxn_smarts = '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#X-[*:2]'.replace('X', new_atom)

    return rxn_smarts


def change_bond_order() -> str:
    """ Returns a SMARTS string to change a bond order """
    choices = ['[*:1]!-[*:2]>>[*:1]-[*:2]', '[*;!H0:1]-[*;!H0:2]>>[*:1]=[*:2]',
               '[*:1]#[*:2]>>[*:1]=[*:2]', '[*;!R;!H1;!H0:1]~[*:2]>>[*:1]#[*:2]']
    probabilities = [0.45, 0.45, 0.05, 0.05]

    return np.random.choice(choices, p=probabilities)


def delete_cyclic_bond() -> str:
    """ Returns a SMARTS string to delete a cyclic bond"""
    return '[*:1]@[*:2]>>([*:1].[*:2])'


def add_ring() -> str:
    """ Returns the SMARTS string to add a ring to a molecule """
    choices = ['[*;!r;!H0:1]~[*;!r:2]~[*;!r;!H0:3]>>[*:1]1~[*:2]~[*:3]1',
               '[*;!r;!H0:1]~[*!r:2]~[*!r:3]~[*;!r;!H0:4]>>[*:1]1~[*:2]~[*:3]~[*:4]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*;!r;!H0:5]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]1',
               '[*;!r;!H0:1]~[*!r:2]~[*:3]~[*:4]~[*!r:5]~[*;!r;!H0:6]>>[*:1]1~[*:2]~[*:3]~[*:4]~[*:5]~[*:6]1']
    probabilities = [0.05, 0.05, 0.45, 0.45]

    return np.random.choice(choices, p=probabilities)


def change_atom(mol: Chem.Mol) -> str:
    """ Returns a SMARTS string to change an atom in a molecule to a different one """
    choices = ['#6', '#7', '#8', '#9', '#16', '#17', '#35']
    probabilities = [0.15, 0.15, 0.14, 0.14, 0.14, 0.14, 0.14]

    first = np.random.choice(choices, p=probabilities)
    while not mol.HasSubstructMatch(Chem.MolFromSmarts('[' + first + ']')):
        first = np.random.choice(choices, p=probabilities)
    second = np.random.choice(choices, p=probabilities)
    while second == first:
        second = np.random.choice(choices, p=probabilities)

    return '[X:1]>>[Y:1]'.replace('X', first).replace('Y', second)


def mutate(mol: Chem.Mol) -> Union[None, Chem.Mol]:
    """ Mutates a molecule based on actions

    :param mol: the molecule to mutate
    :param mutation_rate: the mutation rate
    :param molecule_options: any filters that should be applied
    :returns: A valid mutated molecule or none if it was not possible
    """
    Chem.Kekulize(mol, clearAromaticFlags=True)
    probabilities = [0.15, 0.14, 0.14, 0.14, 0.14, 0.14, 0.15]
    for i in range(10):
        rxn_smarts_list = 7 * ['']
        rxn_smarts_list[0] = insert_atom()
        rxn_smarts_list[1] = change_bond_order()
        rxn_smarts_list[2] = delete_cyclic_bond()
        rxn_smarts_list[3] = add_ring()
        rxn_smarts_list[4] = delete_atom()
        rxn_smarts_list[5] = change_atom(mol)
        rxn_smarts_list[6] = append_atom()
        rxn_smarts = np.random.choice(rxn_smarts_list, p=probabilities)

        rxn = AllChem.ReactionFromSmarts(rxn_smarts)

        new_mol_trial = rxn.RunReactants((mol,))

        new_molecules: List[Chem.Mol] = []
        for m in new_mol_trial:
            m = m[0]
            if mol_ok(m) and ring_ok(m):
                new_molecules.append(m)

        if len(new_molecules) > 0:
            return np.random.choice(new_molecules)

    return None

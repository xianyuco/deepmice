import numpy as np
from openbabel import pybel


xscore_model_vdw = [
    1.90,
    1.80,
    1.70,
    1.50,
    2.10,
    2.00,
    1.20,
    1.90,
    1.80,
    1.80,
    1.80,
    1.80,
    1.80,
    1.80,
    1.80,
    1.70,
    1.70,
    1.70,
    1.70,
    1.70,
    1.70,
    1.70,
    2.00,
    2.00,
    2.00,
    1.80,
    1.80,
    2.00,
    2.20,
]


ob_model_vdw = np.array([
    1.70,
    1.55,
    1.52,
    1.47,
    1.80,
    1.80,
    1.20,
    1.70,
    1.55,
    1.55,
    1.55,
    1.55,
    1.55,
    1.55,
    1.55,
    1.52,
    1.52,
    1.52,
    1.52,
    1.52,
    1.52,
    1.52,
    1.80,
    1.80,
    1.80,
    1.80,
    1.75,
    1.83,
    1.98,
], dtype=np.float32)



neighbor_type_dict = {
    'C': 0,
    'O': 1,
    'N': 2,
    'CC': 3,
    'CO': 4,
    'CN': 5,
    'NN': 6,
    'CCC': 7,
    'CCN': 8,
    'CNO': 9,
    'CCO': 10,
    'COO': 11,
    'CCF': 12,
    'CNN': 13,
    'NNN': 14,
    'other': 15
}


def whichrestype(atom):
    atom = atom if not isinstance(atom, pybel.Atom) else atom.OBAtom
    return atom.GetResidue().GetName() if atom.GetResidue() is not None else None


def is_hydrophobic_atom(atm: pybel.Atom):
    if atm.atomicnum == 6 and set([natom.GetAtomicNum() for natom
                                   in pybel.ob.OBAtomAtomIter(atm.OBAtom)]).issubset({1, 6}):
        return True
    else:
        return False


def is_hba(atm: pybel.Atom):
    if atm.OBAtom.IsHbondAcceptor() and atm.atomicnum not in [9, 17, 35, 53]:
        return True
    else:
        return False


def is_hbd(atm: pybel.Atom):
    if atm.OBAtom.IsHbondDonor() or atm.OBAtom.IsHbondDonorH():
        return True
    if is_hydrophobic_atom(atm) and len([a for a in pybel.ob.OBAtomAtomIter(atm.OBAtom) if a.GetAtomicNum() == 1]) > 0:
        return True
    return False


def is_hal(atm: pybel.Atom):
    if atm.atomicnum in [8, 7, 16]:
        n_atoms = [na for na in pybel.ob.OBAtomAtomIter(atm.OBAtom) if na.GetAtomicNum() in [6, 7, 15, 16]]
        if len(n_atoms) == 1:
            return True
    return False


def charged_positive_negative(atm: pybel.Atom):
    if whichrestype(atm.OBAtom) is not None and whichrestype(atm.OBAtom).upper() in (
            'ARG', 'HIS', 'LYS'):
        if atm.OBAtom.GetType().startswith('N') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
            return 'positive'
        else:
            return None
    if whichrestype(atm.OBAtom) is not None and whichrestype(atm.OBAtom).upper() in (
            'GLU', 'ASP'):
        if atm.OBAtom.GetType().startswith('O') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
            return 'negative'
        else:
            return None
    return None


def is_metal_binding(atm: pybel.Atom):
    if whichrestype(atm.OBAtom) is not None and whichrestype(atm.OBAtom).upper() in ['ASP', 'GLU', 'SER', 'THR', 'TYR'] and atm.OBAtom.GetType().startswith(
            'O') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
        return True
    if whichrestype(atm.OBAtom) is not None and whichrestype(atm.OBAtom).upper() == 'HIS' and atm.OBAtom.GetType().startswith(
            'N') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
        return True
    if whichrestype(atm.OBAtom) is not None and whichrestype(atm.OBAtom).upper() == 'CYS' and atm.OBAtom.GetType().startswith(
            'S') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
        return True
    if atm.OBAtom.GetResidue() is not None:
        if atm.OBAtom.GetType().startswith('O') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 2) and whichrestype(
                atm.OBAtom).upper() != 'HOH':
            return True
    return False

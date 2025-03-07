import numpy as np


openbabel_vdw_radii = {
    'H': 1.10,
    'He': 1.40,
    'Li': 1.81,
    'Be': 1.53,
    'B': 1.92,
    'C': 1.70,
    'N': 1.55,
    'O': 1.52,
    'F': 1.47,
    'Ne': 1.54,
    'Na': 2.27,
    'Mg': 1.73,
    'Al': 1.84,
    'Si': 2.10,
    'P': 1.80,
    'S': 1.80,
    'Cl': 1.75,
    'Ar': 1.88,
    'K': 2.75,
    'Ca': 2.31,
    'Sc': 2.30,
    'Ti': 2.15,
    'V': 2.05,
    'Cr': 2.05,
    'Mn': 2.05,
    'Fe': 2.05,
    'Co': 2.00,
    'Ni': 2.00,
    'Cu': 2.00,
    'Zn': 2.10,
    'Ga': 1.87,
    'Ge': 2.11,
    'As': 1.85,
    'Se': 1.90,
    'Br': 1.83,
    'Kr': 2.02,
    'Rb': 3.03,
    'Sr': 2.49,
    'Y': 2.40,
    'Zr': 2.30,
    'Nb': 2.15,
    'Mo': 2.10,
    'Tc': 2.05,
    'Ru': 2.05,
    'Rh': 2.00,
    'Pd': 2.05,
    'Ag': 2.10,
    'Cd': 2.20,
    'In': 2.20,
    'Sn': 1.93,
    'Sb': 2.17,
    'Te': 2.06,
    'I': 1.98,
    'Xe': 2.16,
    'Cs': 3.43,
    'Ba': 2.68,
    'La': 2.50,
    'Ce': 2.48,
    'Pr': 2.47,
    'Nd': 2.45,
    'Pm': 2.43,
    'Sm': 2.42,
    'Eu': 2.40,
    'Gd': 2.38,
    'Tb': 2.37,
    'Dy': 2.35,
    'Ho': 2.33,
    'Er': 2.32,
    'Tm': 2.30,
    'Yb': 2.28,
    'Lu': 2.27,
    'Hf': 2.25,
    'Ta': 2.20,
    'W': 2.10,
    'Re': 2.05,
    'Os': 2.00,
    'Ir': 2.00,
    'Pt': 2.05,
    'Au': 2.10,
    'Hg': 2.05,
    'Tl': 1.96,
    'Pb': 2.02,
    'Bi': 2.07,
    'Po': 1.97,
    'At': 2.02,
    'Rn': 2.20,
    'Fr': 3.48,
    'Ra': 2.83,
    'Ac': 2.00,
    'Th': 2.40,
    'Pa': 2.00,
    'U': 2.30,
    'Np': 2.00,
    'Pu': 2.00,
    'Am': 2.00,
    'Cm': 2.00,
    'Bk': 2.00,
    'Cf': 2.00,
    'Es': 2.00,
    'Fm': 2.00,
    'Md': 2.00,
    'No': 2.00,
    'Lr': 2.00,
    'Rf': 2.00,
    'Db': 2.00,
    'Sg': 2.00,
    'Bh': 2.00,
    'Hs': 2.00,
    'Mt': 2.00,
    'Ds': 2.00,
    'Rg': 2.00,
    'Cn': 2.00,
    'Nh': 2.00,
    'Fl': 2.00,
    'Mc': 2.00,
    'Lv': 2.00,
    'Ts': 2.00,
    'Og': 2.00,
    'other': 2.00
}


def get_vdw_matrix(mol_a, mol_b):
    symbols = list(openbabel_vdw_radii.keys())

    def get_kind(mol):
        mol_kind = []
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            if sym not in symbols:
                sym = 'other'
            mol_kind.append(sym)
        return mol_kind

    mol_a_kind = get_kind(mol_a)
    mol_b_kind = get_kind(mol_b)
    vdw_radius_matrix = []
    for a_kind in mol_a_kind:  # noqa
        tmp = []
        for b_kind in mol_b_kind:
            ra = openbabel_vdw_radii.get(a_kind)
            rb = openbabel_vdw_radii.get(b_kind)
            if ra is None:
                ra = 1.4
            if rb is None:
                rb = 1.4
            tmp.append((ra + rb) * 0.55)
        vdw_radius_matrix.append(tmp)
    return np.array(vdw_radius_matrix, dtype=np.float32)

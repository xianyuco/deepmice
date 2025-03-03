from openbabel import pybel

from .utils import (
    is_hydrophobic_atom,
    is_hba,
    is_hbd,
    is_hal,
    charged_positive_negative,
    is_metal_binding,
    neighbor_type_dict
)


def get_mol_features(mol):
    atoms_type = []
    neighbors_type = []
    for obatom in pybel.ob.OBMolAtomIter(mol.OBMol):
        atom = pybel.Atom(obatom)
        if atom.atomicnum == 1:
            continue
        x1 = atom.atomicnum

        is_hydrophobic = [False, True]
        ishydrophobic = is_hydrophobic.index(is_hydrophobic_atom(atom))
        x11 = ishydrophobic

        is_hbacceptor = [False, True]
        ishba = is_hbacceptor.index(is_hba(atom))
        x12 = ishba

        is_hbdonor = [False, True]
        ishbd = is_hbdonor.index(is_hbd(atom))
        x13 = ishbd

        is_halogen = [False, True]
        ishal = is_halogen.index(is_hal(atom))
        x14 = ishal

        is_charged_pos_neg = ['positive', 'negative']
        try:
            charged_pos_neg = is_charged_pos_neg.index(charged_positive_negative(atom))
        except Exception:
            charged_pos_neg = 2
        x15 = charged_pos_neg

        is_metalbind = [False, True]
        ismetalbinding = is_metalbind.index(is_metal_binding(atom))
        x16 = ismetalbinding
        if x1 == 6 and x11 == 0:
            atoms_type.append(0)

        elif x1 == 7 and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(1)

        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(2)

        elif x1 == 9 or x1 == 17 or x1 == 35 or x1 == 53:
            atoms_type.append(3)

        elif x1 == 15:
            atoms_type.append(4)

        elif (x1 == 16) and (x12 == 0) and (x14 == 0) and (x16 == 0):
            atoms_type.append(5)

        elif (3 <= x1) and (x1 <= 4):
            atoms_type.append(6)
        elif (11 <= x1) and (x1 <= 13):
            atoms_type.append(6)
        elif (19 <= x1) and (x1 < 32):
            atoms_type.append(6)
        elif (37 <= x1) and (x1 < 51):
            atoms_type.append(6)
        elif (55 <= x1) and (x1 < 85):
            atoms_type.append(6)
        elif (87 <= x1) and (x1 < 109):
            atoms_type.append(6)

        elif (x1 == 6) and (x11 == 1):
            atoms_type.append(7)

        elif (x1 == 7) and (x12 == 0) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(8)

        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(9)

        elif (x1 == 7) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(10)
        elif (x1 == 7) and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(10)
        elif (x1 == 7) and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 0) and (x16 == 0):
            atoms_type.append(10)
        elif (x1 == 7) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(10)
        elif (x1 == 7) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 0) and (x16 == 0):
            atoms_type.append(10)
        elif (x1 == 7) and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 0) and (x16 == 1):
            atoms_type.append(10)
        elif (x1 == 7) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 0) and (x16 == 1):
            atoms_type.append(10)

        elif (x1 == 7) and (x12 == 1) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(11)

        elif (x1 == 7) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(12)
        elif (x1 == 7) and (x12 == 0) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(12)
        elif (x1 == 7) and (x12 == 0) and (x13 == 1) and (x14 == 0) and (x15 == 0) and (x16 == 0):
            atoms_type.append(12)
        elif (x1 == 7) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(12)
        elif (x1 == 7) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 0) and (x16 == 0):
            atoms_type.append(12)
        elif (x1 == 7) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 0) and (x16 == 1):
            atoms_type.append(12)

        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(13)
        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(13)
        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 0) and (x16 == 0):
            atoms_type.append(13)
        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(13)
        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 0) and (x16 == 0):
            atoms_type.append(13)
        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 0) and (x16 == 1):
            atoms_type.append(13)
        elif (x1 == 7) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 0) and (x16 == 1):
            atoms_type.append(13)

        elif (x1 == 7) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(14)
        elif (x1 == 7) and (x12 == 1) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(14)
        elif (x1 == 7) and (x12 == 1) and (x13 == 1) and (x14 == 0) and (x15 == 0) and (x16 == 0):
            atoms_type.append(14)
        elif (x1 == 7) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(14)
        elif (x1 == 7) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 0) and (x16 == 0):
            atoms_type.append(14)
        elif (x1 == 7) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 0) and (x16 == 1):
            atoms_type.append(14)

        elif (x1 == 8) and (x12 == 0) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(15)

        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(16)

        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(17)
        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(17)
        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 1) and (x16 == 0):
            atoms_type.append(17)
        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(17)
        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 1) and (x16 == 0):
            atoms_type.append(17)
        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 0) and (x15 == 1) and (x16 == 1):
            atoms_type.append(17)
        elif (x1 == 8) and (x12 == 0) and (x13 == 0) and (x14 == 1) and (x15 == 1) and (x16 == 1):
            atoms_type.append(17)

        elif (x1 == 8) and (x12 == 1) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 0):
            atoms_type.append(18)

        elif (x1 == 8) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(19)
        elif (x1 == 8) and (x12 == 0) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(19)
        elif (x1 == 8) and (x12 == 0) and (x13 == 1) and (x14 == 0) and (x15 == 1) and (x16 == 0):
            atoms_type.append(19)
        elif (x1 == 8) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(19)
        elif (x1 == 8) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 1) and (x16 == 0):
            atoms_type.append(19)
        elif (x1 == 8) and (x12 == 0) and (x13 == 1) and (x14 == 1) and (x15 == 1) and (x16 == 1):
            atoms_type.append(19)

        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(20)
        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(20)
        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 1) and (x16 == 0):
            atoms_type.append(20)
        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(20)
        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 1) and (x16 == 0):
            atoms_type.append(20)
        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 1) and (x15 == 1) and (x16 == 1):
            atoms_type.append(20)
        elif (x1 == 8) and (x12 == 1) and (x13 == 0) and (x14 == 0) and (x15 == 1) and (x16 == 1):
            atoms_type.append(20)

        elif (x1 == 8) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 0):
            atoms_type.append(21)
        elif (x1 == 8) and (x12 == 1) and (x13 == 1) and (x14 == 0) and (x15 == 2) and (x16 == 1):
            atoms_type.append(21)
        elif (x1 == 8) and (x12 == 1) and (x13 == 1) and (x14 == 0) and (x15 == 1) and (x16 == 0):
            atoms_type.append(21)
        elif (x1 == 8) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 2) and (x16 == 1):
            atoms_type.append(21)
        elif (x1 == 8) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 1) and (x16 == 0):
            atoms_type.append(21)
        elif (x1 == 8) and (x12 == 1) and (x13 == 1) and (x14 == 1) and (x15 == 1) and (x16 == 1):
            atoms_type.append(21)

        elif (x1 == 16) and (x12 == 1) and (x14 == 0) and (x16 == 0):
            atoms_type.append(22)

        elif (x1 == 16) and (x12 == 0) and (x14 == 1) and (x16 == 0):
            atoms_type.append(23)
        elif (x1 == 16) and (x12 == 0) and (x14 == 0) and (x16 == 1):
            atoms_type.append(23)
        elif (x1 == 16) and (x12 == 0) and (x14 == 1) and (x16 == 1):
            atoms_type.append(23)

        elif (x1 == 16) and (x12 == 1) and (x14 == 1) and (x16 == 0):
            atoms_type.append(24)
        elif (x1 == 16) and (x12 == 1) and (x14 == 0) and (x16 == 1):
            atoms_type.append(24)
        elif (x1 == 16) and (x12 == 1) and (x14 == 1) and (x16 == 1):
            atoms_type.append(24)
        else:
            atoms_type.append(25)
        rkey = []
        for na in pybel.ob.OBAtomAtomIter(obatom):
            na = pybel.Atom(na)
            num = na.atomicnum
            if num == 1:
                continue
            s = '-'
            if num == 1:
                s = 'H'
            elif num == 6 or num == 14:
                s = 'C'
            elif num == 7 or num == 15:
                s = 'N'
            elif num == 8 or num == 16:
                s = 'O'
            elif num == 9 or num == 17 or num == 35 or num == 53:
                s = 'F'
            rkey.append(s)
        rkey = sorted(rkey)
        key = ''
        for i in rkey:
            key += str(i)
        if neighbor_type_dict.get(key) is None:
            key = 'other'
        neighbors_type.append(neighbor_type_dict[key])
    return atoms_type, neighbors_type


def get_atoms_features(obmol):
    obmol.removeh()
    obmol.addh()

    atoms_type, neighbors_type = get_mol_features(obmol)
    obmol.removeh()
    return atoms_type, neighbors_type

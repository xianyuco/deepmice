# This file is modified from https://github.com/sc8668/RTMScore
# MIT License Copyright (c) 2023 sc8668

import re
from itertools import permutations
import warnings

from scipy.spatial import distance_matrix
import numpy as np
from openbabel import pybel
from openbabel import openbabel as ob
from openbabel.pybel import Atom
from rdkit import Chem
from rdkit.Chem import AllChem
import torch as th
import dgl

import MDAnalysis as mda
from MDAnalysis.analysis import distances

warnings.filterwarnings('ignore')
ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
pybel.ob.cvar.obErrorLog.SetOutputLevel(0)


METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',
         "CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", "PR", "ND",
         "GD", "TB", "DY", "ER",
         "TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]
RES_MAX_NATOMS = 24

atom_origin_type_map = {
    'C': 0, 'N': 1, 'O': 2, 'S': 3, 'F': 4, 'P': 5, 'Cl': 6,
    'Br': 7, 'I': 8, 'B': 9, 'Si': 10, 'Fe': 11, 'Zn': 12,
    'Cu': 13, 'Mn': 14, 'Mo': 15, 'other': 16
}


def prot_to_graph(prot, cutoff, lpos=None,
                  prot_torsions_sin_cos=None,
                  rigidgroups_gt_frames=None,
                  protein=None):
    u = mda.Universe(prot)
    g = dgl.DGLGraph()

    num_residues = len(u.residues)
    g.add_nodes(num_residues)

    res_feats = np.array([calc_res_features(res) for res in u.residues])
    fdim = res_feats.shape[1]
    if prot_torsions_sin_cos is not None:
        fdim += prot_torsions_sin_cos.shape[1]
    if rigidgroups_gt_frames is not None:
        fdim += rigidgroups_gt_frames.shape[1]
    rfeats = th.zeros((res_feats.shape[0], fdim), dtype=th.float32)

    res_feats = th.tensor(res_feats)
    rfeats[:, :res_feats.shape[1]] = res_feats
    cdim = res_feats.shape[1]
    if prot_torsions_sin_cos is not None:
        rfeats[:prot_torsions_sin_cos.shape[0], cdim:cdim+prot_torsions_sin_cos.shape[1]] = th.tensor(prot_torsions_sin_cos)

        cdim += prot_torsions_sin_cos.shape[1]
    if rigidgroups_gt_frames is not None:
        rfeats[:rigidgroups_gt_frames.shape[0], cdim:cdim+rigidgroups_gt_frames.shape[1]] = th.tensor(rigidgroups_gt_frames)

    g.ndata["feats"] = rfeats
    edgeids, distm = obatin_edge(u, cutoff)
    src_list, dst_list = zip(*edgeids)
    g.add_edges(src_list, dst_list)

    g.ndata["ca_pos"] = th.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    g.ndata["center_pos"] = th.tensor(u.atoms.center_of_mass(compound='residues'))
    dis_matx_ca = distance_matrix(g.ndata["ca_pos"], g.ndata["ca_pos"])
    cadist = th.tensor([dis_matx_ca[i, j] for i, j in edgeids]) * 0.1
    dis_matx_center = distance_matrix(g.ndata["center_pos"], g.ndata["center_pos"])
    cedist = th.tensor([dis_matx_center[i, j] for i, j in edgeids]) * 0.1
    edge_connect = th.tensor(np.array([check_connect(u, x, y) for x, y in zip(src_list, dst_list)]))
    g.edata["feats"] = th.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), th.tensor(distm)],
                              dim=1)
    g.ndata.pop("ca_pos")
    g.ndata.pop("center_pos")
    positions = protein.atom_positions
    atom_types = protein.atom_types
    mask = protein.atom_mask

    aas = np.array([])
    protein_atom_type = []
    min_dis = []
    num = 0
    for res in u.residues:
        if num < positions.shape[0] and protein.aatype[num] != 20:
            t1 = positions[num]
            m1 = mask[num]
            tmp = []
            tmp_type = []
            for i in range(t1.shape[0]):
                if m1[i] > 0.5:
                    tmp.append(t1[i])
                    aidx = atom_origin_type_map[atom_types[num][i]]
                    avec = np.zeros(len(atom_origin_type_map), dtype=np.int32)
                    avec[aidx] = 1
                    tmp_type.append(avec)
            a1 = np.array(tmp)
            tmp_type = np.array(tmp_type)
            if lpos is not None:
                min_dis.append(distance_matrix(a1, lpos).min())
            a2 = np.full((RES_MAX_NATOMS - a1.shape[0], 3), np.nan)
            tmp_type_2 = np.full((RES_MAX_NATOMS - a1.shape[0], len(atom_origin_type_map)), np.nan)
        else:

            if lpos is not None:
                min_dis.append(distance_matrix(res.atoms.positions, lpos).min())
            if len(res.atoms) > 24:
                a1 = res.atoms[:24].positions
            else:
                a1 = res.atoms.positions
            if len(res.atoms) > 24:
                a2 = np.full((0, 3), np.nan)
                tmp_type_2 = np.full((0, len(atom_origin_type_map)), np.nan)
            else:
                a2 = np.full((RES_MAX_NATOMS - len(res.atoms), 3), np.nan)
                tmp_type_2 = np.full((RES_MAX_NATOMS - len(res.atoms), len(atom_origin_type_map)), np.nan)

            tmp_type = []
            for aaa_idx, aaa in enumerate(res.atoms):
                if aaa_idx > 23:
                    break
                try:
                    aidx = atom_origin_type_map[aaa.element]
                except Exception as e:
                    print(e)
                    aidx = len(atom_origin_type_map) - 1
                avec = np.zeros(len(atom_origin_type_map), dtype=np.int32)
                avec[aidx] = 1
                tmp_type.append(avec)
            tmp_type = np.array(tmp_type)

        aa = np.concatenate([a1, a2], axis=0)
        protein_atom_type.append(np.concatenate([tmp_type, tmp_type_2], axis=0))
        if num == 0:
            aas = np.array([aa])
        else:
            aas = np.vstack((aas, np.array([aa])))
        num += 1
    g.ndata["pos"] = th.tensor(np.array(aas))
    g.ndata["atom_type"] = th.tensor(np.array(protein_atom_type))

    if lpos is not None:
        return g, th.tensor(min_dis)
    else:
        return g


def obtain_ca_pos(res):
    if obtain_resname(res) == "M":
        return res.atoms.positions[0]
    else:
        try:
            pos = res.atoms.select_atoms("name CA").positions[0]
            return pos
        except Exception:
            return res.atoms.positions.mean(axis=0)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def obtain_self_dist(res):
    try:

        xx = res.atoms
        dists = distances.self_distance_array(xx.positions)
        ca = xx.select_atoms("name CA")
        c = xx.select_atoms("name C")
        n = xx.select_atoms("name N")
        o = xx.select_atoms("name O")
        return [dists.max() * 0.1, dists.min() * 0.1, distances.dist(ca, o)[-1][0] * 0.1,
                distances.dist(o, n)[-1][0] * 0.1, distances.dist(n, c)[-1][0] * 0.1]
    except Exception:
        return [0, 0, 0, 0, 0]


def obtain_dihediral_angles(res):
    try:
        if res.phi_selection() is not None:
            phi = res.phi_selection().dihedral.value()
        else:
            phi = 0
        if res.psi_selection() is not None:
            psi = res.psi_selection().dihedral.value()
        else:
            psi = 0
        if res.omega_selection() is not None:
            omega = res.omega_selection().dihedral.value()
        else:
            omega = 0
        if res.chi1_selection() is not None:
            chi1 = res.chi1_selection().dihedral.value()
        else:
            chi1 = 0
        return [phi * 0.01, psi * 0.01, omega * 0.01, chi1 * 0.01]
    except Exception:
        return [0, 0, 0, 0]


def calc_res_features(res):
    return np.array(one_of_k_encoding_unk(obtain_resname(res),
                                          ['GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'PRO', 'PHE', 'TYR',
                                           'TRP', 'SER', 'THR', 'CYS', 'MET', 'ASN', 'GLN', 'ASP',
                                           'GLU', 'LYS', 'ARG', 'HIS', 'MSE', 'CSO', 'PTR', 'TPO',
                                           'KCX', 'CSD', 'SEP', 'MLY', 'PCA', 'LLP', 'M', 'X']) +
                    obtain_self_dist(res)
                    + obtain_dihediral_angles(res)
                    )


def obtain_resname(res):
    if res.resname[:2] == "CA":
        resname = "CA"
    elif res.resname[:2] == "FE":
        resname = "FE"
    elif res.resname[:2] == "CU":
        resname = "CU"
    else:
        resname = res.resname.strip()

    if resname in METAL:
        return "M"
    else:
        return resname


def obatin_edge(u, cutoff=10.0):
    cutoff = 10.0
    edgeids = []
    dismin = []
    dismax = []
    for res1, res2 in permutations(u.residues, 2):
        dist = calc_dist(res1, res2)
        if dist.min() <= cutoff:
            edgeids.append([res1.ix, res2.ix])
            dismin.append(dist.min() * 0.1)
            dismax.append(dist.max() * 0.1)
    return edgeids, np.array([dismin, dismax]).T


def check_connect(u, i, j):
    if abs(i - j) != 1:
        return 0
    else:
        if i > j:
            i = j
        nb1 = len(u.residues[i].get_connections("bonds"))
        nb2 = len(u.residues[i + 1].get_connections("bonds"))
        nb3 = len(u.residues[i:i + 2].get_connections("bonds"))
        if nb1 + nb2 == nb3 + 1:
            return 1
        else:
            return 0


def calc_dist(res1, res2):
    dist_array = distances.distance_array(res1.atoms.positions, res2.atoms.positions)
    return dist_array


def whichrestype(atom):
    atom = atom if not isinstance(atom, Atom) else atom.OBAtom
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
            return 'unk'
    if whichrestype(atm.OBAtom) is not None and whichrestype(atm.OBAtom).upper() in (
            'GLU', 'ASP'):
        if atm.OBAtom.GetType().startswith('O') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
            return 'negative'
        else:
            return 'unk'
    return 'unk'


def is_metal_binding(atm: pybel.Atom):
    if whichrestype(atm.OBAtom) is not None and whichrestype(atm.OBAtom).upper() in [
            'ASP', 'GLU', 'SER', 'THR',
            'TYR'] and atm.OBAtom.GetType().startswith('O') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
        return True
    if whichrestype(atm.OBAtom) is not None and whichrestype(
        atm.OBAtom).upper() == 'HIS' and atm.OBAtom.GetType().startswith(
            'N') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
        return True
    if whichrestype(atm.OBAtom) is not None and whichrestype(
        atm.OBAtom).upper() == 'CYS' and atm.OBAtom.GetType().startswith(
            'S') and atm.OBAtom.GetResidue().GetAtomProperty(atm.OBAtom, 8):
        return True
    if atm.OBAtom.GetResidue() is not None:
        if atm.OBAtom.GetType().startswith('O') and atm.OBAtom.GetResidue().GetAtomProperty(
                atm.OBAtom,
                2) and whichrestype(
                atm.OBAtom).upper() != 'HOH':
            return True
    return False


def get_neighbors_type_num(atm: pybel.Atom):
    ATOM_CODES = {}
    atom_classes = [(6, 'C'), (7, 'N'), (8, 'O'), (9, 'F')]
    for code, (atomidx, name) in enumerate(atom_classes):
        if type(atomidx) is list:
            for a in atomidx:
                ATOM_CODES[a] = code
        else:
            ATOM_CODES[atomidx] = code
    atom_neighbors_type_num = [0, 0, 0, 0, 0]
    for a in pybel.ob.OBAtomAtomIter(atm.OBAtom):
        try:
            a = pybel.Atom(a)
            anum = a.atomicnum
            if anum == 16:
                anum = 8
            if anum == 15:
                anum = 7
            if anum == 17 or anum == 35 or anum == 53:
                anum = 9
            classes = ATOM_CODES[anum]
        except Exception:
            classes = 4
        atom_neighbors_type_num[classes] = atom_neighbors_type_num[classes] + 1
    neigCount = sum(atom_neighbors_type_num)
    neiglabel = 15
    if neigCount == 1:
        if atom_neighbors_type_num[0] == 1:
            neiglabel = 0
        if atom_neighbors_type_num[2] == 1:
            neiglabel = 1
        if atom_neighbors_type_num[1] == 1:
            neiglabel = 2
    elif neigCount == 2:
        if atom_neighbors_type_num[0] == 2:
            neiglabel = 3
        if atom_neighbors_type_num[0] == 1 and atom_neighbors_type_num[2] == 1:
            neiglabel = 4
        if atom_neighbors_type_num[0] == 1 and atom_neighbors_type_num[1] == 1:
            neiglabel = 5
        if atom_neighbors_type_num[1] == 2:
            neiglabel = 6
    elif neigCount == 3:
        if atom_neighbors_type_num[0] == 3:
            neiglabel = 7
        if atom_neighbors_type_num[0] == 2 and atom_neighbors_type_num[1] == 1:
            neiglabel = 8
        if atom_neighbors_type_num[0] == 1 and atom_neighbors_type_num[1] == 1 and atom_neighbors_type_num[2] == 1:
            neiglabel = 9
        if atom_neighbors_type_num[0] == 2 and atom_neighbors_type_num[2] == 1:
            neiglabel = 10
        if atom_neighbors_type_num[0] == 1 and atom_neighbors_type_num[2] == 2:
            neiglabel = 11
        if atom_neighbors_type_num[0] == 2 and atom_neighbors_type_num[3] == 1:
            neiglabel = 12
        if atom_neighbors_type_num[0] == 1 and atom_neighbors_type_num[1] == 2:
            neiglabel = 13
        if atom_neighbors_type_num[1] == 3:
            neiglabel = 14
    else:
        neiglabel = 15
    return one_of_k_encoding(
        neiglabel,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])


def calc_bond_stereo(mol: pybel.Molecule, bond):
    facade = ob.OBStereoFacade(mol.OBMol)
    bid = bond.GetId()
    HasTetrahedralStereo = facade.HasTetrahedralStereo(bid)
    HasCisTransStereo = facade.HasCisTransStereo(bid)
    HasSquarePlanarStereo = facade.HasSquarePlanarStereo(bid)
    if not HasTetrahedralStereo and not HasCisTransStereo and not HasSquarePlanarStereo:
        return [0, 0, 0, 1]
    else:
        return [facade.HasTetrahedralStereo(bid), facade.HasCisTransStereo(bid), facade.HasSquarePlanarStereo(bid), 0]


def calc_atom_features101(atom,

                          explicit_H=False):

    results = one_of_k_encoding_unk(
        atom.atomicnum,
        [6, 7, 8, 16, 9, 15, 17, 35, 53, 5, 14, 26, 30, 29, 25, 42, 10000]) \
        + one_of_k_encoding(
        atom.OBAtom.GetTotalDegree(),
        [0, 1, 2, 3, 4, 5, 6]) \
        + [atom.formalcharge, atom.OBAtom.GetTotalValence()] \
        + one_of_k_encoding_unk(atom.hyb, [1, 2, 3, 4, 5, 6, 10000]) + [atom.OBAtom.IsAromatic(),
                                                                        atom.OBAtom.IsChiral()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(
            atom.OBAtom.GetImplicitHCount() + atom.OBAtom.ExplicitHydrogenCount(),
            [0, 1, 2, 3, 4])

    results = results + calc_atom_features_plip(atom) + get_neighbors_type_num(atom)
    return np.array(results)


def calc_atom_features(atom, explicit_H=False):
    results = one_of_k_encoding_unk(
        atom.atomicnum,
        [6, 7, 8, 16, 9, 15, 17, 35, 53, 5, 14, 26, 30, 29, 25, 42, 10000]) \
        + one_of_k_encoding(
            is_hydrophobic_atom(atom),
        [True, False]) \
        + one_of_k_encoding(
            is_hba(atom),
        [True, False]) \
        + one_of_k_encoding(
            is_hbd(atom),
        [True, False]) \
        + one_of_k_encoding(
            is_hydrophobic_atom(atom),
        [True, False]) \
        + one_of_k_encoding(
            is_hal(atom),
        [True, False]) \
        + one_of_k_encoding(
            charged_positive_negative(atom),
        ['positive', 'negative', 'unk']) \
        + one_of_k_encoding(
            is_metal_binding(atom),
        [True, False]) \
        + one_of_k_encoding(
            atom.OBAtom.IsAromatic(),
        [True, False]) \
        + one_of_k_encoding(
            atom.OBAtom.IsChiral(),
        [True, False])

    results = results + get_neighbors_type_num(atom)
    return np.array(results)


def calc_atom_features_plip(atom):
    x1 = atom.atomicnum
    x11 = is_hydrophobic_atom(atom)
    x12 = is_hba(atom)
    x13 = is_hbd(atom)
    x14 = is_hal(atom)
    x15 = charged_positive_negative(atom)
    x16 = is_metal_binding(atom)
    pliptype = 44
    if x1 == 6 and not (x11):
        pliptype = 0
    elif (x1 == 7) and (not x12) and (not x13) and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 1
    elif (x1 == 8) and (not x12) and (not x13) and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 2
    elif (x1 == 16) and (not x12) and (not x13) and (not x14) and (not x16):
        pliptype = 3
    elif (x1 == 9) and (not x13):
        pliptype = 4
    elif (x1 == 15) and (not x12) and (not x13):
        pliptype = 5
    elif (x1 == 17) and (not x13):
        pliptype = 6
    elif (x1 == 35) and (not x13):
        pliptype = 7
    elif (x1 == 53) and (not x13):
        pliptype = 8
    elif (x1 == 5):
        pliptype = 9
    elif (x1 == 14):
        pliptype = 10
    elif (x1 == 26):
        pliptype = 11
    elif (x1 == 30):
        pliptype = 12
    elif (x1 == 29):
        pliptype = 13
    elif (x1 == 25):
        pliptype = 14
    elif (x1 == 42):
        pliptype = 15
    elif (x1 == 6) and x11 and (not x12) and (not x13):
        pliptype = 16
    elif (x1 == 6) and (not x11) and (x12) and (not x13):
        pliptype = 17
    elif (x1 == 6) and (not x11) and (not x12) and x13:
        pliptype = 18
    elif (x1 == 6) and x11 and x12 and (not x13):
        pliptype = 19
    elif (x1 == 6) and x11 and (not x12) and x13:
        pliptype = 20
    elif (x1 == 7) and (not x12) and x13 and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 21
    elif (x1 == 7) and x12 and (not x13) and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 22
    elif (x1 == 7) and (not x12) and (not x13) and x14 and (x15 == "unk") and (not x16):
        pliptype = 23
    elif (x1 == 7) and (not x12) and (not x13) and (not x14) and (x15 == "unk") and x16:
        pliptype = 23
    elif (x1 == 7) and (not x12) and (not x13) and (not x14) and (x15 == "positive") and (not x16):
        pliptype = 23
    elif (x1 == 7) and (not x12) and (not x13) and x14 and (x15 == "unk") and x16:
        pliptype = 23
    elif (x1 == 7) and (not x12) and (not x13) and x14 and (x15 == "positive") and (not x16):
        pliptype = 23
    elif (x1 == 7) and (not x12) and (not x13) and (not x14) and (x15 == "positive") and x16:
        pliptype = 23
    elif (x1 == 7) and (not x12) and (not x13) and x14 and (x15 == "positive") and x16:
        pliptype = 23
    elif (x1 == 7) and x12 and x13 and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 24
    elif (x1 == 7) and (not x12) and x13 and x14 and (x15 == "unk") and (not x16):
        pliptype = 25
    elif (x1 == 7) and (not x12) and x13 and (not x14) and (x15 == "unk") and x16:
        pliptype = 25
    elif (x1 == 7) and (not x12) and x13 and (not x14) and (x15 == "positive") and (not x16):
        pliptype = 25
    elif (x1 == 7) and (not x12) and x13 and x14 and (x15 == "unk") and x16:
        pliptype = 25
    elif (x1 == 7) and (not x12) and x13 and x14 and (not x15) and (not x16):
        pliptype = 25
    elif (x1 == 7) and (not x12) and x13 and x14 and (x15 == "positive") and x16:
        pliptype = 25
    elif (x1 == 7) and (not x12) and x13 and (not x14) and (x15 == "positive") and x16:
        pliptype = 25
    elif (x1 == 7) and x12 and (not x13) and x14 and (x15 == "unk") and (not x16):
        pliptype = 26
    elif (x1 == 7) and x12 and (not x13) and (not x14) and (x15 == "unk") and x16:
        pliptype = 26
    elif (x1 == 7) and x12 and (not x13) and (not x14) and (x15 == "positive") and (not x16):
        pliptype = 26
    elif (x1 == 7) and x12 and (not x13) and x14 and (x15 == "unk") and x16:
        pliptype = 26
    elif (x1 == 7) and x12 and (not x13) and x14 and (x15 == "positive") and not x16:
        pliptype = 26
    elif (x1 == 7) and x12 and (not x13) and x14 and (x15 == "positive") and x16:
        pliptype = 26
    elif (x1 == 7) and x12 and (not x13) and (not x14) and (x15 == "positive") and x16:
        pliptype = 26
    elif (x1 == 7) and x12 and x13 and x14 and (x15 == "unk") and (not x16):
        pliptype = 27
    elif (x1 == 7) and x12 and x13 and (not x14) and (x15 == "unk") and x16:
        pliptype = 27
    elif (x1 == 7) and x12 and x13 and (not x14) and (x15 == "positive") and (not x16):
        pliptype = 27
    elif (x1 == 7) and x12 and x13 and x14 and (x15 == "unk") and x16:
        pliptype = 27
    elif (x1 == 7) and x12 and x13 and x14 and (x15 == "positive") and (not x16):
        pliptype = 27
    elif (x1 == 7) and x12 and x13 and x14 and (x15 == "positive") and x16:
        pliptype = 27
    elif (x1 == 8) and (not x12) and x13 and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 28
    elif (x1 == 8) and x12 and (not x13) and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 29
    elif (x1 == 8) and (not x12) and (not x13) and x14 and (x15 == "unk") and (not x16):
        pliptype = 30
    elif (x1 == 8) and (not x12) and (not x13) and (not x14) and (x15 == "unk") and x16:
        pliptype = 30
    elif (x1 == 8) and (not x12) and (not x13) and (not x14) and (x15 == "negative") and (not x16):
        pliptype = 30
    elif (x1 == 8) and (not x12) and (not x13) and x14 and (x15 == "unk") and x16:
        pliptype = 30
    elif (x1 == 8) and (not x12) and (not x13) and x14 and (x15 == "negative") and (not x16):
        pliptype = 30
    elif (x1 == 8) and (not x12) and (not x13) and (not x14) and (x15 == "negative") and x16:
        pliptype = 30
    elif (x1 == 8) and (not x12) and (not x13) and x14 and (x15 == "negative") and x16:
        pliptype = 30
    elif (x1 == 8) and x12 and x13 and (not x14) and (x15 == "unk") and (not x16):
        pliptype = 31
    elif (x1 == 8) and (not x12) and x13 and x14 and (x15 == "unk") and (not x16):
        pliptype = 32
    elif (x1 == 8) and (not x12) and x13 and (not x14) and (x15 == "unk") and x16:
        pliptype = 32
    elif (x1 == 8) and (not x12) and x13 and (not x14) and (x15 == "negative") and (not x16):
        pliptype = 32
    elif (x1 == 8) and (not x12) and x13 and x14 and (x15 == "unk") and x16:
        pliptype = 32
    elif (x1 == 8) and (not x12) and x13 and x14 and (x15 == "negative") and (not x16):
        pliptype = 32
    elif (x1 == 8) and (not x12) and x13 and x14 and (x15 == "negative") and x16:
        pliptype = 32
    elif (x1 == 8) and (not x12) and x13 and (not x14) and (x15 == "negative") and x16:
        pliptype = 32
    elif (x1 == 8) and x12 and (not x13) and x14 and (x15 == "unk") and (not x16):
        pliptype = 33
    elif (x1 == 8) and x12 and (not x13) and (not x14) and (x15 == "unk") and x16:
        pliptype = 33
    elif (x1 == 8) and x12 and (not x13) and (not x14) and (x15 == "negative") and (not x16):
        pliptype = 33
    elif (x1 == 8) and x12 and (not x13) and x14 and (x15 == "unk") and x16:
        pliptype = 33
    elif (x1 == 8) and x12 and (not x13) and x14 and (x15 == "negative") and (not x16):
        pliptype = 33
    elif (x1 == 8) and x12 and (not x13) and x14 and (x15 == "negative") and x16:
        pliptype = 33
    elif (x1 == 8) and x12 and (not x13) and (not x14) and (x15 == "negative") and x16:
        pliptype = 33
    elif (x1 == 8) and x12 and x13 and x14 and (x15 == "unk") and (not x16):
        pliptype = 34
    elif (x1 == 8) and x12 and x13 and (not x14) and (x15 == "unk") and x16:
        pliptype = 34
    elif (x1 == 8) and x12 and x13 and (not x14) and (x15 == "negative") and (not x16):
        pliptype = 34
    elif (x1 == 8) and x12 and x13 and x14 and (x15 == "unk") and x16:
        pliptype = 34
    elif (x1 == 8) and x12 and x13 and x14 and (x15 == "negative") and (not x16):
        pliptype = 34
    elif (x1 == 8) and x12 and x13 and x14 and (x15 == "negative") and x16:
        pliptype = 34
    elif (x1 == 16) and x12 and (not x14) and (not x16):
        pliptype = 35
    elif (x1 == 16) and (not x12) and x14 and (not x16):
        pliptype = 36
    elif (x1 == 16) and (not x12) and (not x14) and x16:
        pliptype = 36
    elif (x1 == 16) and (not x12) and x14 and x16:
        pliptype = 36
    elif (x1 == 16) and x12 and x14 and (not x16):
        pliptype = 37
    elif (x1 == 16) and x12 and (not x14) and x16:
        pliptype = 37
    elif (x1 == 16) and x12 and x14 and x16:
        pliptype = 37

    elif (x1 == 9) and x13:
        pliptype = 38
    elif (x1 == 15) and x12 and (not x13):
        pliptype = 39
    elif (x1 == 15) and (not x12) and x13:
        pliptype = 40
    elif (x1 == 17) and x13:
        pliptype = 41
    elif (x1 == 35) and x13:
        pliptype = 42
    elif (x1 == 53) and x13:
        pliptype = 43
    else:
        pliptype = 44

    return one_of_k_encoding(
        pliptype,
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
         30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44])


def calc_bond_features(bond, mol):
    bt = bond.GetBondOrder()
    possible_bonds_type = [1, 2, 3]
    if bt not in possible_bonds_type:
        bt = 10000
    bond_feats = [
        bt == 1, bt == 2,
        bt == 3, bt == 10000, bond.IsAromatic(),
        bond.IsInRing()] + calc_bond_stereo(mol, bond)
    return np.array(bond_feats).astype(int)


def load_mol_ob(molpath, explicit_H=False, use_chirality=True):

    if re.search(r'.pdb$', molpath):
        mol = next(pybel.readfile("pdb", molpath))
    elif re.search(r'.mol2$', molpath):
        mol = next(pybel.readfile("mol2", molpath))
    elif re.search(r'.sdf$', molpath):
        mol = next(pybel.readfile("sdf", molpath))
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    if not explicit_H:
        mol.removeh()
    return mol


def load_mol_rd(molpath, explicit_H=False, use_chirality=True):

    if re.search(r'.pdb$', molpath):
        mol = Chem.MolFromPDBFile(molpath, removeHs=not explicit_H)
    elif re.search(r'.mol2$', molpath):
        mol = Chem.MolFromMol2File(molpath, removeHs=not explicit_H)
    elif re.search(r'.sdf$', molpath):
        mol = Chem.MolFromMolFile(molpath, removeHs=not explicit_H)
    else:
        raise IOError("only the molecule files with .pdb|.sdf|.mol2 are supported!")

    if use_chirality:
        Chem.AssignStereochemistryFrom3D(mol)
    return mol


def mol_to_graph(mol, explicit_H=False, use_chirality=True):
    g = dgl.DGLGraph()
    num_atoms = mol.OBMol.NumAtoms()
    g.add_nodes(num_atoms)

    atom_feats = np.array([
        calc_atom_features101(a, explicit_H=explicit_H)
        for a in mol.atoms
    ])

    g.ndata["atom"] = th.tensor(atom_feats)
    atomCoords = np.array([[atm.OBAtom.GetX(), atm.OBAtom.GetY(), atm.OBAtom.GetZ()] for atm in mol.atoms])
    g.ndata["pos"] = th.tensor(atomCoords)

    src_list = []
    dst_list = []
    bond_feats_all = []
    num_bonds = mol.OBMol.NumBonds()
    for i in range(num_bonds):
        bond = mol.OBMol.GetBond(i)
        u = bond.GetBeginAtomIdx() - 1
        v = bond.GetEndAtomIdx() - 1
        bond_feats = calc_bond_features(bond, mol)
        src_list.extend([u, v])
        dst_list.extend([v, u])
        bond_feats_all.append(bond_feats)
        bond_feats_all.append(bond_feats)

    g.add_edges(src_list, dst_list)
    g.edata["bond"] = th.tensor(np.array(bond_feats_all))
    return g


def mol2word(mol, radius):
    info = {}
    _ = AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)

    mol_atoms = [a.GetIdx() for a in mol.GetAtoms()]
    dict_atoms = {x: None for x in mol_atoms}

    for element in info:
        for atom_idx, radius_at in info[element]:
            if radius_at == 0:
                continue
            dict_atoms[atom_idx] = element
    return dict_atoms


def mol_to_graph_lig(prot_path, lig_path,
                     cutoff=10.0, explicit_H=False, use_chirality=True):
    lig = load_mol_ob(lig_path, explicit_H=explicit_H, use_chirality=use_chirality)
    gl = mol_to_graph(lig, explicit_H=explicit_H, use_chirality=use_chirality)
    return gl

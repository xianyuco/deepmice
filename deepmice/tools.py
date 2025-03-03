import os
import random
import logging

import numpy as np

from openbabel import openbabel
from openbabel import pybel
from rdkit.Chem.Lipinski import RotatableBondSmarts
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.PDBIO import PDBIO, Select

from .tree import LigandSegment
from .xpdb import SloppyStructureBuilder


ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
pybel.ob.obErrorLog.StopLogging()


logger = logging.getLogger('deepmice_logger')


class Box:

    def __init__(self, center_x, center_y, center_z, size_x, size_y, size_z) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z
        self.size_x = size_x
        self.size_y = size_y
        self.size_z = size_z
        self.box = np.array(
            [center_x, center_y, center_z, size_x, size_y, size_z],
            dtype=np.float32)
        self.center = self.box[:3]
        self.size = self.box[3:]
        self.range = np.stack([
            self.center - self.size / 2,
            self.center + self.size / 2])
        self.move_size = np.array([0, 0, 0], dtype=np.float32)
        self._origin_box = None

    @property
    def origin_box(self):
        '''将当前box转化为左下角坐标放置在原点的box'''
        if self._origin_box is None:
            move_size = self.center - self.size / 2
            center = self.center - move_size
            tmp_box = Box(*center, *self.size)
            tmp_box.move_size = move_size
            self._origin_box = tmp_box
        return self._origin_box


def generate_random_file_path(sufix='.pdb'):
    while True:
        tmp_path = os.path.join('/tmp', str(random.randint(10000000, 100000000)) + sufix)
        if not os.path.exists(tmp_path):
            break
    return tmp_path


def ob_read_file(file_path):
    if not os.path.exists(file_path):
        logger.info('%s 文件不存在' % file_path)
        return None
    suffix = os.path.splitext(file_path)[-1]
    try:
        if suffix == '.mol2':
            mol = next(pybel.readfile('mol2', file_path))
        elif suffix == '.mol':
            mol = next(pybel.readfile('mol', file_path))
        elif suffix == '.pdb':
            mol = next(pybel.readfile('pdb', file_path))
        elif suffix == '.pdbqt':
            mol = next(pybel.readfile('pdbqt', file_path))
        else:
            mol = None
    except Exception as e:
        logger.warning(e)
        mol = None
    return mol


def ob_read_sdf_file(file_path):
    try:
        mol_iter = pybel.readfile('sdf', file_path)
    except Exception as e:
        logger.warning(e)
        mol_iter = None
    return mol_iter


def calc_rmsd(pos, all_pos):
    '''

    Args:
        pos: shape=[num_atom, 3]
        all_pos: shape=[num_pos, num_atom, 3]
    '''

    if len(all_pos.shape) == 2:
        all_pos = all_pos[np.newaxis, :, :]
    tmp = np.sum((pos[np.newaxis, :, :] - all_pos) ** 2, (1, 2))
    return np.squeeze(np.sqrt(tmp / all_pos.shape[1]))


def ob_calc_rmsd(ori_path, new_path):
    '''ori_path为sdf文件时，可多对一计算rmsd，返回一个list'''

    tmp_path = os.path.join('/tmp', str(random.randint(0, 10000000)) + '_rms.txt')
    os.system('obrms {} {} > {}'.format(ori_path, new_path, tmp_path))
    if os.path.exists(tmp_path):
        with open(tmp_path, 'r') as f:
            lines = f.readlines()
        result = []
        for line in lines:
            t = line.strip().split(' ')[-1]
            if t == 'inf':
                t = None
            else:
                try:
                    t = float(t)
                except Exception:
                    t = None
            result.append(t)
        if len(result) == 1:
            result = result[0]
        elif len(result) == 0:
            result = None
        os.remove(tmp_path)
    else:
        result = None
    return result


def remove_h2o(pdb_path, save_path):
    if os.path.splitext(pdb_path)[1] != '.pdb':
        logger.info('%s 文件后缀名不是pdb， 无法执行去水操作。' % pdb_path)
        return False
    with open(pdb_path) as f:
        lines = f.readlines()
    res_lines = []
    h2o_indexes = set()
    for line in lines:
        if line[:6] == 'HETATM' and line[17: 20] == 'HOH':
            h2o_indexes.add(line[6:11])
            continue
        if line[:6] == 'CONECT':
            idx = line[6: 11]
            if idx in h2o_indexes:
                continue
        res_lines.append(line)
    with open(save_path, 'w') as f:
        f.writelines(res_lines)
    return True


def IsImide(querybond: openbabel.OBBond):
    if querybond.GetBondOrder() != 2:
        return False
    bgn = querybond.GetBeginAtom()
    end = querybond.GetEndAtom()
    if ((bgn.GetAtomicNum() == 6 and end.GetAtomicNum() == 7)
            or (bgn.GetAtomicNum() == 7 and end.GetAtomicNum() == 6)):
        return True
    return False


def IsAmidine(querybond: openbabel.OBBond):
    c = n = None
    bgn = querybond.GetBeginAtom()
    end = querybond.GetEndAtom()
    if bgn.GetAtomicNum() == 6 and end.GetAtomicNum() == 7:
        c = bgn
        n = end
    if bgn.GetAtomicNum() == 7 and end.GetAtomicNum() == 6:
        c = end
        n = bgn
    if c is None or n is None:
        return False
    if querybond.GetBondOrder() != 1:
        return False
    if n.GetTotalDegree() != 3:
        return False

    for bond in openbabel.OBAtomBondIter(c):
        if IsImide(bond):
            return True
    return False


def IsRotatableBonds(bond: openbabel.OBBond):
    if bond.GetBondOrder() != 1 or bond.IsAromatic() or bond.IsAmide() or IsAmidine(bond) or bond.IsInRing():
        return False
    if ((bond.GetBeginAtom()).GetExplicitDegree() == 1) or ((bond.GetEndAtom()).GetExplicitDegree() == 1):
        return False
    return True


def get_ob_rotatable_bond_idx(obmol):
    rot_atom_pairs = []
    rot_bond_set = []
    if obmol.OBMol.NumBonds() > 0:
        for bond in openbabel.OBMolBondIter(obmol.OBMol):
            if IsRotatableBonds(bond):
                rot_bond_set.append(bond.GetIdx())
                rot_atom_pairs.append((bond.GetBeginAtom().GetIdx() - 1, bond.GetEndAtom().GetIdx() - 1))
    return rot_atom_pairs, rot_bond_set


def get_rdkit_rotatable_bond_idx(mol):
    rot_atom_pairs = mol.GetSubstructMatches(RotatableBondSmarts)
    rot_bond_set = list(set([mol.GetBondBetweenAtoms(*i).GetIdx() for i in rot_atom_pairs]))
    return rot_atom_pairs, rot_bond_set


def split_rdkit_mol_atom(rdkit_mol_atom, bond):
    left_atoms_idx = []
    right_atoms_idx = []
    begin_atom = bond.GetBeginAtom()
    left_atom_idx = begin_atom.GetIdx()
    end_atom = bond.GetEndAtom()
    right_atom_idx = end_atom.GetIdx()

    def get_nei(atom, atom_indexes):
        nei = atom.GetNeighbors()
        for i in nei:
            idx = i.GetIdx()
            if idx != left_atom_idx and idx != right_atom_idx and idx not in atom_indexes:
                atom_indexes.append(idx)
                get_nei(i, atom_indexes)

    get_nei(begin_atom, left_atoms_idx)
    get_nei(end_atom, right_atoms_idx)

    if len(left_atoms_idx) > len(right_atoms_idx):
        left_atoms_idx, right_atoms_idx = right_atoms_idx, left_atoms_idx
        left_atom_idx, right_atom_idx = right_atom_idx, left_atom_idx
    if len(left_atoms_idx) + len(right_atoms_idx) + 2 < rdkit_mol_atom.GetNumAtoms():
        total_atoms_idx = list(range(rdkit_mol_atom.GetNumAtoms()))
        for i in total_atoms_idx:
            if (i != left_atom_idx
                    and i != right_atom_idx
                    and i not in left_atoms_idx
                    and i not in right_atoms_idx):
                right_atoms_idx.append(i)
    return left_atom_idx, right_atom_idx, left_atoms_idx, right_atoms_idx


def get_rdkit_rotatable_bond_info(rdkit_mol, bond_idx):
    res = []
    for idx in bond_idx:
        res.append(split_rdkit_mol_atom(rdkit_mol, rdkit_mol.GetBondWithIdx(idx)))
    return res


def split_ob_mol_atom(ob_mol_atom, bond):
    left_atoms_idx = []
    right_atoms_idx = []
    begin_atom = bond.GetBeginAtom()
    left_atom_idx = begin_atom.GetIdx() - 1
    end_atom = bond.GetEndAtom()
    right_atom_idx = end_atom.GetIdx() - 1

    def get_nei(atom, atom_indexes):
        for i in openbabel.OBAtomAtomIter(atom):
            idx = i.GetIdx() - 1
            if idx != left_atom_idx and idx != right_atom_idx and idx not in atom_indexes:
                atom_indexes.append(idx)
                get_nei(i, atom_indexes)

    get_nei(begin_atom, left_atoms_idx)
    get_nei(end_atom, right_atoms_idx)

    if len(left_atoms_idx) > len(right_atoms_idx):
        left_atoms_idx, right_atoms_idx = right_atoms_idx, left_atoms_idx
        left_atom_idx, right_atom_idx = right_atom_idx, left_atom_idx
    if len(left_atoms_idx) + len(right_atoms_idx) + 2 < ob_mol_atom.OBMol.NumAtoms():
        total_atoms_idx = list(range(ob_mol_atom.OBMol.NumAtoms()))
        for i in total_atoms_idx:
            if (i != left_atom_idx
                    and i != right_atom_idx
                    and i not in left_atoms_idx
                    and i not in right_atoms_idx):
                right_atoms_idx.append(i)
    return left_atom_idx, right_atom_idx, left_atoms_idx, right_atoms_idx


def get_ob_rotatable_bond_info(ob_mol, bond_idx):
    res = []
    for idx in bond_idx:
        res.append(split_ob_mol_atom(ob_mol, ob_mol.OBMol.GetBond(idx)))
    return res


def get_rdkit_tree(mol, rot_bond_index, positions, start_idx=None):
    num_atoms = mol.GetNumAtoms()
    added_atom_indexes = []
    added_bond_indexes = []
    free_degree_start_index = 0

    def _gen_tree(left_atom_idx, out_tree):
        if left_atom_idx not in added_atom_indexes:
            out_tree.add_atom(left_atom_idx)
            added_atom_indexes.append(left_atom_idx)
            if len(added_atom_indexes) == num_atoms:
                return

        left_atom = mol.GetAtomWithIdx(left_atom_idx)
        bonds = left_atom.GetBonds()

        for bond in bonds:
            if bond.GetIdx() in added_bond_indexes:
                continue
            batom = bond.GetBeginAtom()
            eatom = bond.GetEndAtom()
            if batom.GetIdx() != left_atom_idx:
                next_atom = batom
            else:
                next_atom = eatom

            added_bond_indexes.append(bond.GetIdx())
            if bond.GetIdx() in rot_bond_index:
                if batom.GetIdx() == left_atom_idx:
                    child_axis_begin = batom.GetIdx()
                    child_axis_end = eatom.GetIdx()
                else:
                    child_axis_begin = eatom.GetIdx()
                    child_axis_end = batom.GetIdx()
                nonlocal free_degree_start_index
                free_degree_start_index += 1
                new_tree = LigandSegment(
                    child_axis_end, child_axis_begin, child_axis_end, free_degree_start_index)
                out_tree.add_child(new_tree)
                _gen_tree(next_atom.GetIdx(), new_tree)
            else:
                _gen_tree(next_atom.GetIdx(), out_tree)

    if start_idx is None:
        start_idx = len(positions) // 2
    free_degree_start_index += 5
    tree = LigandSegment(start_idx, None, None, free_degree_start_index, is_root_segment=True)
    tree.set_positions(positions)
    _gen_tree(start_idx, tree)
    tree.tree_info.num_freedegree = free_degree_start_index + 1

    return tree


def get_ob_tree(obmol, rot_bond_index, multiple_positions, start_idx=None):
    num_atoms = len(obmol.atoms)
    added_atom_indexes = []
    added_bond_indexes = []
    free_degree_start_index = 0

    def _gen_tree(left_atom_idx, out_tree):
        if left_atom_idx not in added_atom_indexes:
            out_tree.add_atom(left_atom_idx)
            added_atom_indexes.append(left_atom_idx)
            if len(added_atom_indexes) == num_atoms:
                return

        left_atom = obmol.OBMol.GetAtom(left_atom_idx + 1)
        bonds = [i for i in openbabel.OBAtomBondIter(left_atom)]

        for bond in bonds:
            if bond.GetIdx() in added_bond_indexes:
                continue
            batom = bond.GetBeginAtom()
            eatom = bond.GetEndAtom()
            if (batom.GetIdx() - 1) != left_atom_idx:
                next_atom = batom
            else:
                next_atom = eatom

            added_bond_indexes.append(bond.GetIdx())
            if bond.GetIdx() in rot_bond_index:
                if (batom.GetIdx() - 1) == left_atom_idx:
                    child_axis_begin = batom.GetIdx() - 1
                    child_axis_end = eatom.GetIdx() - 1
                else:
                    child_axis_begin = eatom.GetIdx() - 1
                    child_axis_end = batom.GetIdx() - 1
                nonlocal free_degree_start_index
                free_degree_start_index += 1
                new_tree = LigandSegment(
                    child_axis_end, child_axis_begin, child_axis_end, free_degree_start_index)
                out_tree.add_child(new_tree)
                _gen_tree(next_atom.GetIdx() - 1, new_tree)
            else:
                _gen_tree(next_atom.GetIdx() - 1, out_tree)

    if start_idx is None:
        start_idx = len(multiple_positions[0]) // 2
    tree = LigandSegment(start_idx, None, None, free_degree_start_index, is_root_segment=True)
    tree.set_tree_info(multiple_positions, len(rot_bond_index) + 1)
    _gen_tree(start_idx, tree)
    assert(len(rot_bond_index) == free_degree_start_index)

    return tree


def _get_tree_depth(tree, parent_depth):
    depth = parent_depth + 1
    max_depth = depth
    for c in tree.children:
        child_depth = _get_tree_depth(c, depth)
        if child_depth > max_depth:
            max_depth = child_depth
    return max_depth


def get_tree_depth(tree):
    return _get_tree_depth(tree, 0)


class CutProteinSelect(Select):

    def __init__(self, box: Box):
        self.box = box

    def accept_residue(self, residue):
        for atom in residue.get_list():
            if np.all(atom.coord > self.box.range[0]) and\
               np.all(atom.coord < self.box.range[1]):
                return True
        return False


def cut_protein_with_box(protein_path: str, box: Box, save_pdb_path: str):
    if protein_path.endswith('pdb'):
        parser = PDBParser(PERMISSIVE=True, structure_builder=SloppyStructureBuilder())
    elif protein_path.endswith('cif'):
        parser = MMCIFParser()
    name = os.path.splitext(os.path.split(protein_path)[-1])[0]
    structure = parser.get_structure(name, protein_path)
    select = CutProteinSelect(box)
    io = PDBIO()
    io.set_structure(structure)
    io.save(save_pdb_path, select)


def cut_protein_with_ligand(protein_path: str, ligand_path: str, save_pdb_path: str, extend_length=5):
    mol = ob_read_file(ligand_path)
    if mol is None:
        return
    coords = np.array([a.coords for a in mol.atoms])
    min_coord = coords.min(0)
    max_coord = coords.max(0)
    size = max_coord - min_coord
    center = min_coord + size / 2
    size += extend_length * 2
    box = Box(*center, *size)
    cut_protein_with_box(protein_path, box, save_pdb_path)

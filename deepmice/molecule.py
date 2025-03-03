import os
import copy
import logging

import numpy as np
from openbabel import openbabel
from openbabel import pybel

from .tools import (
    generate_random_file_path,
    ob_read_file,
    ob_read_sdf_file,
    remove_h2o,
    get_ob_rotatable_bond_idx,
    get_ob_rotatable_bond_info,
    get_ob_tree,
    get_tree_depth  # noqa: F401
)


logger = logging.getLogger('deepmice_logger')


class Molecule:

    def __init__(self, removeh=True) -> None:
        self._removeh = removeh
        self._obmol = None
        self._file_path = None
        self._multiple_positions = []
        self._rotatable_bond_info = None
        self._rotatable_bond_index = None
        self._mol_tree = None

    def _is_same_mol(self, mol_a, mol_b):
        a_atoms = mol_a.atoms
        b_atoms = mol_b.atoms
        if len(a_atoms) != len(b_atoms):
            return False
        for a, b in zip(a_atoms, b_atoms):
            if a.atomicnum != b.atomicnum:
                return False
        return True

    @property
    def num_conformers(self):
        return len(self._multiple_positions)

    @property
    def num_atoms(self):
        if len(self._multiple_positions) != 0:
            return len(self._multiple_positions[0])
        return 0

    @property
    def positions(self):
        if len(self._multiple_positions) != 0:
            return self._multiple_positions[0]

    def add_conformer_from_file(self, file_path):
        if file_path[-4:] == '.sdf':
            total_positions = []
            mol_iter = ob_read_sdf_file(file_path)
            if mol_iter is None:
                return False
            first_mol = None
            for mol in mol_iter:
                if self._removeh:
                    mol.removeh()
                if first_mol is None:
                    first_mol = mol
                    if self._obmol is not None:
                        if not self._is_same_mol(self._obmol, first_mol):
                            return False
                    else:
                        self._obmol = first_mol
                else:
                    if not self._is_same_mol(first_mol, mol):
                        return False
                positions = []
                for a in mol.atoms:
                    if a.atomicnum != 1:
                        positions.append(a.coords)
                positions = np.array(positions, dtype=np.float32)
                total_positions.append(positions)
                self._multiple_positions.extend(total_positions)
            return True
        else:
            mol = ob_read_file(file_path)
            return self.add_conformer_from_mol(mol)


    def add_conformer_from_mol(self, obmol):
        if obmol is None:
            return False
        if self._removeh:
            obmol.removeh()
        if self._obmol is None:
            self._obmol = obmol
        else:
            if not self._is_same_mol(self._obmol, obmol):
                return False
        positions = []
        for a in obmol.atoms:
            positions.append(a.coords)
        positions = np.array(positions, dtype=np.float32)
        self._multiple_positions.append(positions)
        return True

    @property
    def rotatable_bond(self):
        if self._rotatable_bond_info is not None:
            return self._rotatable_bond_info
        self._rotatable_bond_info = get_ob_rotatable_bond_info(
            self._obmol, self.rotatable_bond_index)
        return self._rotatable_bond_info

    @property
    def rotatable_bond_index(self):
        if self._rotatable_bond_index is None:
            _, self._rotatable_bond_index = get_ob_rotatable_bond_idx(self._obmol)
        return self._rotatable_bond_index

    @property
    def covalent_bond_index(self):
        u_index = []
        v_index = []
        for bond in openbabel.OBMolBondIter(self._obmol.OBMol):
            u_index.append(bond.GetBeginAtom().GetIdx() - 1)
            v_index.append(bond.GetEndAtom().GetIdx() - 1)
        u_index = np.array(u_index, np.int64)
        v_index = np.array(v_index, np.int64)
        return u_index, v_index

    @property
    def mol_tree(self):
        if self._mol_tree is None:
            positions = self._multiple_positions[0]
            idx = np.argmin(np.sqrt(((positions - positions.mean(0)) ** 2).sum(-1))).tolist()
            self._mol_tree = get_ob_tree(
                self._obmol, self.rotatable_bond_index, self._multiple_positions, idx)
        return self._mol_tree

    def save(self, positions, save_path, score=None):
        positions = np.array(positions)
        save_path_suffix = os.path.splitext(save_path)[-1][1:]
        out = pybel.Outputfile(save_path_suffix, save_path, overwrite=True)
        need_removeh = False
        if len(positions.shape) == 2:
            for atom, i in zip(self._obmol.atoms, positions.tolist()):
                if atom.atomicnum != 1:
                    atom.OBAtom.SetVector(openbabel.vector3(i[0], i[1], i[2]))
                else:
                    need_removeh = True
            out.write(self._obmol)
        elif len(positions.shape) == 3:
            for pos in positions:
                for atom, i in zip(self._obmol.atoms, pos.tolist()):
                    if atom.atomicnum != 1:
                        atom.OBAtom.SetVector(openbabel.vector3(i[0], i[1], i[2]))
                    else:
                        need_removeh = True
                out.write(self._obmol)
        out.close()

        if save_path_suffix == 'pdb' and os.path.exists(save_path):
            with open(save_path, 'r') as f:
                lines = f.readlines()
            pop_index = -1
            for i in range(5):
                if lines[i][: 6] == 'AUTHOR':
                    lines[i] = 'AUTHOR    DeepMice v0.6.5\n'
                elif lines[i][: 6] == 'COMPND':
                    pop_index = i
            if pop_index != -1:
                lines.pop(pop_index)

            if need_removeh:
                try:
                    pop_indexes = []
                    for i in range(len(lines) - 1, -1, -1):
                        if lines[i][:6] == 'ATOM  ' or lines[i][:6] == 'HETATM':
                            if lines[i][76: 78] == ' H':
                                pop_indexes.append(i)
                    for idx in pop_indexes:
                        lines.pop(idx)
                except Exception as e:
                    logger.error('removeh error.\n' + str(e))
            with open(save_path, 'w') as f:
                if score is not None:
                    lines.insert(0, 'REMARK DeepMice DOCK SCORE: {:7.3f}\n'.format(np.round(score, 3)))
                f.writelines(lines)
        elif save_path_suffix == 'sdf' and score is not None and os.path.exists(save_path):
            with open(save_path) as f:
                lines = f.readlines()
            aux = 0
            top_index = -1
            for i, l in enumerate(lines):
                if i == 0 or l[:4] == '$$$$':
                    aux = 2
                    top_index += 1
                    if l[:4] == '$$$$':
                        continue
                if aux == 0:
                    continue
                elif aux == 2:
                    file_name = 'DeepMice_Result_TOP_{}\n'.format(top_index + 1)
                    lines[i] = file_name
                    aux -= 1
                elif aux == 1:
                    if l.strip()[:9] == 'OpenBabel':
                        if score is not None:
                            score_info = '  DeepMice DOCK SCORE: {:7.3f}\n'.format(
                                score[top_index])
                        else:
                            score_info = '  DeepMice DOCK SCORE: ---\n'
                        lines[i] = score_info
                    aux -= 1
            if need_removeh:
                pop_indexes = []
                for i in range(len(lines) - 1, -1, -1):
                    if lines[i][31:32] == 'H':
                        try:
                            float(lines[i][0: 10])
                            float(lines[i][10: 20])
                            float(lines[i][20: 30])
                        except Exception:
                            continue
                        pop_indexes.append(i)
                for idx in pop_indexes:
                    lines.pop(idx)
            with open(save_path, 'w') as f:
                f.writelines(lines)


class Ligand(Molecule):

    def __init__(self, removeh=True) -> None:
        super().__init__(removeh)
        self.move_size = 0

    def moveto_box_center(self, center_point):
        for positions in self._multiple_positions:
            move_size = positions.mean(0) - center_point
            positions = positions - move_size
        return self

    def move(self, move_size):
        self.move_size += move_size
        for positions in self._multiple_positions:
            positions = positions - move_size
        return self

    def save(self, positions, save_path, score=None):
        positions = np.array(positions)
        result_positions = positions + self.move_size
        return super().save(result_positions, save_path, score)

    @property
    def free_degree(self):
        return len(self.rotatable_bond_index) + 6

    @property
    def possible_collision_index(self):
        def get_neighbors(atom_index, rdmol=None, obmol=None):
            if obmol is not None:
                atom = obmol.OBMol.GetAtom(atom_index + 1)
                indexes = [i.GetIdx() - 1 for i in openbabel.OBAtomAtomIter(atom)]
            else:
                atom = rdmol.GetAtomWithIdx(atom_index)
                neighbors = atom.GetNeighbors()
                indexes = [i.GetIdx() for i in neighbors]
            return indexes

        def bonded_to3(rdmol, obmol, atom_index, out, n):
            if atom_index not in out:
                out.append(atom_index)
                if n > 0:
                    neighbors = get_neighbors(atom_index, rdmol, obmol)
                    for neig in neighbors:
                        bonded_to3(rdmol, obmol, neig, out, n-1)

        def bonded_to(rdmol, obmol, atom_index):
            out = []
            bonded_to3(rdmol, obmol, atom_index, out, 3)
            return out

        result_index_u = []
        result_index_v = []
        tmp = set()
        for info in self.rotatable_bond:
            left_idx, right_idx, left_atoms, right_atoms = copy.deepcopy(info)
            left_atoms.append(left_idx)
            right_atoms.append(right_idx)
            for i in left_atoms:
                i_bonded = bonded_to(None, self._obmol, i)
                for j in right_atoms:
                    if j in i_bonded:
                        continue
                    sig = str(i) + '_' + str(j)
                    if sig in tmp:
                        continue
                    result_index_u.append(i)
                    result_index_v.append(j)
                    tmp.update({str(i) + '_' + str(j), str(j) + '_' + str(i)})
        result_index_u = np.array(result_index_u, dtype=np.int64)
        result_index_v = np.array(result_index_v, dtype=np.int64)
        return result_index_u, result_index_v


class Receptor(Molecule):

    def __init__(self, removeh=True, remove_water=True):
        super().__init__(removeh)
        self._remove_water = remove_water
        self._nowater_file_path = None

    def add_conformer_from_file(self, file_path):
        if self._remove_water:
            self._nowater_file_path = generate_random_file_path()
            remove_h2o(file_path, self._nowater_file_path)
            return super().add_conformer_from_file(self._nowater_file_path)
        else:
            return super().add_conformer_from_file(file_path)

    def __del__(self):
        if self._nowater_file_path is not None and os.path.exists(self._nowater_file_path):
            os.remove(self._nowater_file_path)

    @property
    def free_degree(self):
        return 0

    def move(self, move_size):
        self._multiple_positions[0] -= move_size
        return self

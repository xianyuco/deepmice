import os
import logging
import shutil
import tempfile

import numpy as np
import torch

from .molecule import Ligand, Receptor
from .tools import Box, cut_protein_with_box, generate_random_file_path

try:
    from . import csearch
except ModuleNotFoundError:
    from . import csearch_cpu as csearch


logger = logging.getLogger('deepmice_logger')
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler('deepmice.log', 'a', 'utf-8')
file_handler.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.WARNING)
formatter = logging.Formatter("%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


class MLPCScreen:

    def __init__(self, protein_path, box, topk_conformers=10, device='cpu') -> None:
        from .models.mlp.grid_cache import GridCache
        from .models.mlp.prepare_inputs import get_atoms_features

        self.topk_conformers = 1 if topk_conformers < 0 else topk_conformers
        if 'cuda' in device:
            if 'cuda' == device:
                device_index = 0
            else:
                device_index = int(device.split(':')[-1])
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise ValueError('device error.')
        self.device = device

        self.protein_path = protein_path

        self.cut_protein_path = generate_random_file_path()
        box = np.array(box)
        cut_box = Box(*box[:3], *(box[3:] + 16))
        cut_protein_with_box(self.protein_path, cut_box, self.cut_protein_path)

        self.box = Box(*box)
        self.box = self.box.origin_box
        self.move_size = self.box.move_size

        self.protein = Receptor(removeh=True, remove_water=True)
        self.protein.add_conformer_from_file(self.cut_protein_path)
        self.protein.move(self.move_size)

        self.grid_cache = GridCache()
        (self.protein_atoms_type,
         self.protein_neighbors_type) = get_atoms_features(self.protein._obmol)
        cache_dict = self.grid_cache.get_cache()

        self.cs = csearch.MLPCSearch()
        self.cs.set_box(self.box.center_x, self.box.center_y, self.box.center_z,
                        self.box.size_x, self.box.size_y, self.box.size_z)
        self.cs.set_protein_features(
            self.protein_atoms_type,
            self.protein_neighbors_type,
            self.protein.positions)

        self.cs.set_grid_cache(
            self.grid_cache.grid_spacing,
            cache_dict['score'],
            cache_dict['grad'],
            cache_dict['repulsive_energy'],
            cache_dict['repulsive_grad']
        )
        self.cs.init_screen(self.topk_conformers, self.device)

    def run(self, ligand_paths, save_path):
        from .models.mlp.prepare_inputs import get_atoms_features

        if isinstance(ligand_paths, str):
            ligand_paths = [ligand_paths]

        ligand = Ligand(removeh=True)
        for p in ligand_paths:
            status = ligand.add_conformer_from_file(p)
            if status is False:
                logger.warning('add ligand conformer failed: {}'.format(p))
        if ligand.num_conformers == 0:
            err_info = 'no conformer {}'.format(ligand_paths[0])
            logger.error(err_info)
            raise ValueError(err_info)

        (ligand_atoms_type,
         ligand_neighbors_type) = get_atoms_features(ligand._obmol)

        if len(ligand_atoms_type) >= 80:
            logger.error('The number of ligand (%s) atoms exceeds 80.' % ligand_paths[0])
            return
        if (ligand.free_degree >= 30):
            logger.error('The number of ligand (%s) rotatable_bonds exceeds 24.' % ligand_paths[0])
            return
        logger.info('start docking: ligand %s, protein %s.' % (
            ligand_paths[0], self.protein_path))

        ligand.move(self.move_size).moveto_box_center(self.box.center)
        collision_index = np.array(ligand.possible_collision_index).T

        self.cs.build_ligand_tree(ligand.mol_tree)
        self.cs.set_ligand_features(
            ligand_atoms_type,
            ligand_neighbors_type,
            collision_index)

        positions, energies, rmsds = self.cs.run()

        topk_conformers = (
            len(positions) if self.topk_conformers > len(positions)
            else self.topk_conformers
        )
        ligand.save(positions[0], save_path, energies[0])

        if topk_conformers > 1:
            topk_file_path = '{}.sdf'.format(os.path.splitext(save_path)[0])
            ligand.save(positions, topk_file_path, energies)
        logger.info('end docking: ligand %s, protein %s, score %5.2f.' %
                    (ligand_paths[0], self.protein_path, energies[0]))
        return energies

    def run_batch(self, batch_inputs):
        pass

    def __del__(self):
        if os.path.exists(self.cut_protein_path):
            os.remove(self.cut_protein_path)


class MLPCDock:

    def __init__(self, ligand_paths, protein_path, box) -> None:
        from .models.mlp.grid_cache import GridCache
        from .models.mlp.prepare_inputs import get_atoms_features

        if isinstance(ligand_paths, str):
            ligand_paths = [ligand_paths]
        self.ligand_paths = ligand_paths
        self.protein_path = protein_path

        self.cut_protein_path = generate_random_file_path()
        box = np.array(box)
        cut_box = Box(*box[:3], *(box[3:] + 16))
        cut_protein_with_box(self.protein_path, cut_box, self.cut_protein_path)

        self.box = Box(*box)
        self.box = self.box.origin_box
        move_size = self.box.move_size

        logger.info('start docking: ligand %s, protein %s.' % (
            self.ligand_paths[0], self.protein_path))

        self.ligand = Ligand(removeh=True)
        for p in self.ligand_paths:
            status = self.ligand.add_conformer_from_file(p)
            if status is False:
                logger.warning('add ligand conformer failed: {}'.format(p))
        if self.ligand.num_conformers == 0:
            err_info = 'no conformer {}'.format(ligand_paths[0])
            logger.error(err_info)
            raise ValueError(err_info)
        self.ligand.move(move_size).moveto_box_center(self.box.center)

        self.protein = Receptor(removeh=True, remove_water=True)
        self.protein.add_conformer_from_file(self.cut_protein_path)
        self.protein.move(move_size)

        self.grid_cache = GridCache()
        (self.ligand_atoms_type,
         self.ligand_neighbors_type) = get_atoms_features(self.ligand._obmol)
        (self.protein_atoms_type,
         self.protein_neighbors_type) = get_atoms_features(self.protein._obmol)
        self.collision_index = np.array(self.ligand.possible_collision_index).T

    def run(self, save_path, topk_conformers=10, device='cpu'):
        if 'cuda' in device:
            device_index = int(device.split(':')[-1])
            if device_index < 0 or device_index >= torch.cuda.device_count():
                raise ValueError('device error.')

        if len(self.ligand_atoms_type) >= 80:
            logger.error('The number of ligand (%s) atoms exceeds 80.' % self.ligand_paths[0])
            return
        if (self.ligand.free_degree >= 30):
            logger.error('The number of ligand (%s) rotatable_bonds exceeds 24.' % self.ligand_paths[0])
            return
        cs = csearch.MLPCSearch()
        cs.set_box(self.box.center_x, self.box.center_y, self.box.center_z,
                   self.box.size_x, self.box.size_y, self.box.size_z)
        cs.build_ligand_tree(self.ligand.mol_tree)
        cs.set_ligand_features(
            self.ligand_atoms_type,
            self.ligand_neighbors_type,
            self.collision_index)
        cs.set_protein_features(
            self.protein_atoms_type,
            self.protein_neighbors_type,
            self.protein.positions)

        cache_dict = self.grid_cache.get_cache()
        cs.set_grid_cache(self.grid_cache.grid_spacing,
                          cache_dict['score'],
                          cache_dict['grad'],
                          cache_dict['repulsive_energy'],
                          cache_dict['repulsive_grad'])
        cs.init_screen(topk_conformers, device)
        positions, energies, rmsds = cs.run()
        topk_conformers = 1 if topk_conformers < 0 else topk_conformers
        topk_conformers = len(positions) if topk_conformers > len(positions) else topk_conformers

        self.ligand.save(positions[0], save_path, energies[0])

        if topk_conformers > 1:
            topk_file_path = '{}.sdf'.format(os.path.splitext(save_path)[0])
            self.ligand.save(positions, topk_file_path, energies)
        logger.info('end docking: ligand %s, protein %s, score %5.2f.' %
                    (self.ligand_paths[0], self.protein_path, energies[0]))
        if os.path.exists(self.cut_protein_path):
            os.remove(self.cut_protein_path)
        return energies


class MDNV2CScreen:

    def __init__(self, protein_path, box, topk_conformers=10, device='cpu') -> None:
        from .models.mdn_v2.grid_cache import GridCache
        from .models.mdn_v2.prepare_data.extract_pocket_prody_ob import extract_pocket_with_box

        self.topk_conformers = topk_conformers
        self.device = device
        self.protein_path = protein_path
        box = np.array(box)

        cut_box = Box(*box[:3], *(box[3:] + 10))
        self.pocketdir = tempfile.mkdtemp()
        self.pocket_pdb_path = extract_pocket_with_box(protein_path, cut_box, self.pocketdir)
        self.box = Box(*box).origin_box

        self.grid_cache = GridCache(self.device)
        cache_dict = self.grid_cache.get_cache()

        self.protein = Receptor(removeh=True, remove_water=True)
        self.grid_cache.calc_protein_embedding(self.pocket_pdb_path)
        self.protein._multiple_positions.append(self.grid_cache.protein_positions)
        self.protein.move(self.box.move_size)
        self.protein_atom_type = self.grid_cache.protein_atom_type

        self.cs = csearch.MDNV2CSearch()
        self.cs.set_box(self.box.center_x, self.box.center_y, self.box.center_z,
                        self.box.size_x, self.box.size_y, self.box.size_z)

        self.cs.set_protein_features(
            self.protein_atom_type,
            self.protein.positions)

        self.cs.set_base_cache(
            self.grid_cache.grid_spacing,
            cache_dict['repulsive_energy'],
            cache_dict['repulsive_grad']
        )
        self.cs.init_screen(self.topk_conformers, self.device)

    def run(self, ligand_paths, save_path):

        if isinstance(ligand_paths, str):
            ligand_paths = [ligand_paths]

        ligand = Ligand(removeh=True)
        for p in ligand_paths:
            status = ligand.add_conformer_from_file(p)
            if status is False:
                logger.warning('add ligand conformer failed: {}'.format(p))
        if ligand.num_conformers == 0:
            err_info = 'no conformer {}'.format(ligand_paths[0])
            logger.error(err_info)
            raise ValueError(err_info)

        if len(ligand.positions) >= 80:
            logger.error('The number of ligand (%s) atoms exceeds 80.' % ligand_paths[0])
            return
        if (ligand.free_degree >= 30):
            logger.error('The number of ligand (%s) rotatable_bonds exceeds 24.' % ligand_paths[0])
            return

        logger.info('start docking: ligand %s, protein %s.' % (
            ligand_paths[0], self.protein_path))

        self.grid_cache.calc_ligand_embedding(ligand._obmol)
        self.grid_cache.calc_pi_sigma_mu()
        cache_data, ligand_atom_type, ligand_positions =\
            self.grid_cache.calc_ligand_cache()

        ligand_collision_index = np.array(ligand.possible_collision_index).T

        ligand.move(self.box.move_size).moveto_box_center(self.box.center)

        self.cs.build_ligand_tree(ligand.mol_tree)
        self.cs.set_ligand_features(
            ligand_atom_type.cpu().numpy(),
            ligand_collision_index)
        self.cs.set_grid_cache(cache_data.cpu().numpy())

        positions, energies, rmsds = self.cs.run()

        topk_conformers = (
            len(positions) if self.topk_conformers > len(positions)
            else self.topk_conformers
        )
        ligand.save(positions[0], save_path, energies[0])

        if topk_conformers > 1:
            topk_file_path = '{}.sdf'.format(os.path.splitext(save_path)[0])
            ligand.save(positions, topk_file_path, energies)
        logger.info('end docking: ligand %s, protein %s, score %5.2f.' %
                    (ligand_paths[0], self.protein_path, energies[0]))
        return energies

    def __del__(self):
        if os.path.exists(self.pocketdir):
            shutil.rmtree(self.pocketdir)


class CRotateMove:

    def __init__(self, ligand_mol_tree, box):
        self.ligand_mol_tree = ligand_mol_tree
        self.crm = csearch.CRotateMove()
        self.crm.init_data(ligand_mol_tree, box)

    @property
    def inputs(self):
        return self.crm.get_inputs()

    @inputs.setter
    def inputs(self, values):
        assert len(values) == self.ligand_mol_tree.tree_info.num_freedegree - 1
        self.crm.set_inputs(values)

    def mutate_inputs(self):
        self.crm.mutate_inputs()

    def random_inputs(self):
        self.crm.random_inputs()

    def rotate_move(self, inputs_value=None):
        if inputs_value is not None:
            self.inputs = inputs_value
        return self.crm.rotate_move()

    def rotate_move_grad(self, position_grad):
        assert position_grad.shape[0] == self.ligand_mol_tree.tree_info.local_positions.shape[1]
        return self.crm.rotate_move_grad(position_grad)

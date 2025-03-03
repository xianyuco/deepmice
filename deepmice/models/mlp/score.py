import os

import numpy as np
import torch
from scipy.spatial.distance import cdist

from .grid_cache import GridCache, DISTANCE_SPACING, CURMODEL
from .grid_cache import CURDIR
from .prepare_inputs import get_atoms_features

from deepmice.molecule import Ligand, Receptor


class MLPScore:

    def __init__(self) -> None:
        self.grid_cache = GridCache()
        self.cache_info = self.grid_cache.get_cache()
        sd = torch.load(os.path.join(CURDIR, 'saved_models', CURMODEL))
        self.bias = sd['out_linear.bias'].cpu().numpy().tolist()[0]

    def run(self, ligand_path, protein_path,
            include_auxiliary_score=True,
            include_ligand_covalent_bond_score=True,
            include_ligand_noncovalent_bond_score=True,
            bias=True):
        ligand = Ligand(removeh=True)
        ligand.add_conformer_from_file(ligand_path)
        protein = Receptor(removeh=True, remove_water=True)
        protein.add_conformer_from_file(protein_path)
        (ligand_atoms_type,
         ligand_neighbors_type) = get_atoms_features(ligand._obmol)
        (protein_atoms_type,
         protein_neighbors_type) = get_atoms_features(protein._obmol)
        collision_index = ligand.possible_collision_index
        ligand_atoms_type = np.array(ligand_atoms_type)
        ligand_neighbors_type = np.array(ligand_neighbors_type)
        protein_atoms_type = np.array(protein_atoms_type)
        protein_neighbors_type = np.array(protein_neighbors_type)

        lp_distance = cdist(ligand.positions, protein.positions)
        less8_index = np.nonzero(lp_distance < 8)
        less8_distance = lp_distance[less8_index]
        lp_dis_index = np.floor(less8_distance / DISTANCE_SPACING).astype(np.int64)
        lp_dis_weight = (less8_distance - lp_dis_index * DISTANCE_SPACING) / DISTANCE_SPACING

        lp_l_atoms = ligand_atoms_type[less8_index[0]]
        lp_l_nbrs  = ligand_neighbors_type[less8_index[0]]
        lp_p_atoms = protein_atoms_type[less8_index[1]]
        lp_p_nbrs  = protein_neighbors_type[less8_index[1]]

        if include_auxiliary_score:
            lp_score_0 = self.cache_info['score'][lp_l_atoms, lp_l_nbrs, lp_p_atoms, lp_p_nbrs, lp_dis_index]
            lp_score_1 = self.cache_info['score'][lp_l_atoms, lp_l_nbrs, lp_p_atoms, lp_p_nbrs, lp_dis_index + 1]
        else:
            lp_score_0 = self.cache_info['mlp_model_cache'][lp_l_atoms, lp_l_nbrs, lp_p_atoms, lp_p_nbrs, lp_dis_index]
            lp_score_1 = self.cache_info['mlp_model_cache'][lp_l_atoms, lp_l_nbrs, lp_p_atoms, lp_p_nbrs, lp_dis_index + 1]
        lp_score = lp_score_0 * (1 - lp_dis_weight) + lp_score_1 * lp_dis_weight

        if bias:
            score = lp_score.sum() - self.bias
        else:
            score = lp_score.sum()

        ll_distance = None
        if include_ligand_noncovalent_bond_score:
            ll_distance = cdist(ligand.positions, ligand.positions)
            collision_mask = np.zeros(ll_distance.shape, dtype=np.bool_)
            collision_mask[collision_index] = 1
            collision_mask = np.logical_and(collision_mask, ll_distance < 4.2)
            ll_sel_index = np.nonzero(collision_mask)

            ll_sel_distance = ll_distance[ll_sel_index]
            ll_sel_dis_index = np.floor(ll_sel_distance / DISTANCE_SPACING).astype(np.int64)
            ll_sel_dis_weight = (ll_sel_distance - ll_sel_dis_index * DISTANCE_SPACING) / DISTANCE_SPACING

            ll_l0_atoms = ligand_atoms_type[ll_sel_index[0]]
            ll_l1_atoms = ligand_atoms_type[ll_sel_index[1]]

            ll_score_0 = self.cache_info['repulsive_energy'][ll_l0_atoms, ll_l1_atoms, ll_sel_dis_index]
            ll_score_1 = self.cache_info['repulsive_energy'][ll_l0_atoms, ll_l1_atoms, ll_sel_dis_index + 1]
            ll_score = ll_score_0 * (1 - ll_sel_dis_weight) + ll_score_1 * ll_sel_dis_weight

            score += ll_score.sum()

        if include_ligand_covalent_bond_score:
            if ll_distance is None:
                ll_distance = cdist(ligand.positions, ligand.positions)
            covalent_bond_index = ligand.covalent_bond_index
            lc_0_atoms = ligand_atoms_type[covalent_bond_index[0]]
            lc_1_atoms = ligand_atoms_type[covalent_bond_index[1]]
            lc_0_nbrs = ligand_neighbors_type[covalent_bond_index[0]]
            lc_1_nbrs = ligand_neighbors_type[covalent_bond_index[1]]
            lc_distance = ll_distance[covalent_bond_index]
            lc_dis_index = np.floor(lc_distance / DISTANCE_SPACING).astype(np.int64)
            lc_dis_weight = (lc_distance - lc_dis_index * DISTANCE_SPACING) / DISTANCE_SPACING

            lc_score_0 = self.cache_info['mlp_model_cache'][lc_0_atoms, lc_0_nbrs, lc_1_atoms, lc_1_nbrs, lc_dis_index]
            lc_score_1 = self.cache_info['mlp_model_cache'][lc_0_atoms, lc_0_nbrs, lc_1_atoms, lc_1_nbrs, lc_dis_index + 1]
            lc_score = lc_score_0 * (1 - lc_dis_weight) + lc_score_1 * lc_dis_weight
            score += lc_score.sum()

        return score

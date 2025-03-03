import os
import shutil
from os import PathLike
from pathlib import Path
import tempfile
import typing as t

import numpy as np
import torch

from deepmice.tools import Box, ob_read_sdf_file, ob_read_file
from deepmice.molecule import Ligand

from .grid_cache import (
    GridCache,
    DISTANCE_SPACING,
    PL_DISTANCE_THRESHOLD,
    calculate_probablity
)
from .prepare_data.extract_pocket_prody_ob import extract_pocket_with_box


class MDNV2CScore:
    device = 'cpu'
    grid_cache = None

    def __init__(self, device='cuda:0'):
        if not device == MDNV2CScore.device:
            MDNV2CScore.device = device
            self.device = device
            MDNV2CScore.grid_cache = GridCache(self.device)
        else:
            if MDNV2CScore.grid_cache is None:
                MDNV2CScore.grid_cache = GridCache(self.device)
        self.grid_cache = MDNV2CScore.grid_cache

    @staticmethod
    def load_protein_cache(protein_cache_path: PathLike):
        return torch.load(protein_cache_path)

    def generate_protein_cache(self, protein_path: PathLike, cut_box,
                               save_path: t.Optional[PathLike] = None):

        cut_box = Box(*cut_box)
        pocketdir = tempfile.mkdtemp()
        pocket_pdb_path = extract_pocket_with_box(protein_path, cut_box, pocketdir)
        MDNV2CScore.grid_cache.calc_protein_embedding(pocket_pdb_path)
        protein_cache = (
            self.grid_cache.protein_embedding,
            self.grid_cache.protein_positions,
            self.grid_cache.protein_non_nan
        )
        if save_path is not None:
            torch.save(protein_cache, save_path)
        if os.path.exists(pocketdir):
            shutil.rmtree(pocketdir)
        return protein_cache

    def preprocess_protein(self, protein_path: t.Optional[PathLike] = None,
                           cut_box=None,
                           protein_cache_path=None):
        if protein_cache_path is not None and os.path.exists(protein_cache_path):
            protein_cache = self.load_protein_cache(protein_cache_path)
        else:
            protein_cache = self.generate_protein_cache(
                protein_path, cut_box, protein_cache_path
            )

        (self.grid_cache.protein_embedding,
         self.grid_cache.protein_positions,
         self.grid_cache.protein_non_nan) = protein_cache

    def _run_mol(self, mol, use_cache=False) -> t.Optional[float]:
        ligand = Ligand(removeh=True)
        status = ligand.add_conformer_from_mol(mol)
        if not status:
            return None

        self.grid_cache.calc_ligand_embedding(ligand._obmol)
        self.grid_cache.calc_pi_sigma_mu()
        lpdis = torch.cdist(self.grid_cache.ligand_positions,
                            self.grid_cache.protein_positions)
        if use_cache:
            lpdis = lpdis.cpu().numpy()
            (cache_data,
             ligand_atom_type,
             ligand_positions) = self.grid_cache.calc_ligand_cache()
            cache_data *= 1 / self.grid_cache.grid_cache_scale
            cache_data = cache_data.cpu().numpy()

            lpdis_idx = np.round(lpdis / DISTANCE_SPACING).astype(np.int32)
            result_score = 0
            for li in range(lpdis.shape[0]):
                for pi in range(lpdis.shape[1]):
                    if lpdis[li, pi] < PL_DISTANCE_THRESHOLD:
                        result_score += cache_data[li, pi, lpdis_idx[li, pi]]
        else:
            lpdis = lpdis.reshape(-1)
            idx = torch.where(lpdis < PL_DISTANCE_THRESHOLD)
            lpdis = lpdis[idx]

            pi = self.grid_cache.pi[idx]
            sigma = self.grid_cache.sigma[idx]
            mu = self.grid_cache.mu[idx]
            result_score = calculate_probablity(
                pi, sigma, mu, lpdis.view(-1, 1)
            ).sum().item()
        return result_score

    def run(self, ligand_path, use_cache=False) -> t.Optional[float]:
        mol = ob_read_file(ligand_path)
        if mol is None:
            return None

        return self._run_mol(mol, use_cache)

    def run_batch(self,
                  ligand_paths: t.Union[t.List[PathLike], str],
                  use_cache=False) -> t.List[t.Tuple[PathLike, int, float]]:
        if isinstance(ligand_paths, str):
            ligand_paths = [ligand_paths]

        result = []
        for k, ligand_path in enumerate(ligand_paths):
            if Path(ligand_path).suffix == '.sdf':
                mol_iter = ob_read_sdf_file(ligand_path)
            else:
                mol_iter = [ob_read_file(ligand_path)]

            for mol_index, mol in enumerate(mol_iter, start=1):
                try:
                    prob_result = self._run_mol(mol, use_cache)
                except Exception as e:
                    print(f'Score {ligand_path} (index: {mol_index}) error. \n{e}')
                else:
                    if prob_result:
                        result.append((ligand_path, mol_index, prob_result))
        return result

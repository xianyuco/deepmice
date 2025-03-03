import math
import os

import numpy as np
import torch
from torch.distributions import Normal

from .model import MDNV2, DGLGraphTransformer
from .prepare_data.mol2graph_rdmda_ob_res_large import mol_to_graph, prot_to_graph
from .prepare_data.protein import from_pdb_path
from .prepare_data.all_atom import atom37_to_torsion_angles


CURMODEL = 'mdn_v2_graph_largeob.pt'

PL_DISTANCE_THRESHOLD = 5
DISTANCE_SPACING = 0.375

CURDIR = os.path.dirname(os.path.abspath(__file__))
LOADED_MODEL = None
LOADED_GRID_CACHE = None


atom_vdw = np.array([
    1.90,
    1.80,
    1.70,
    2.00,
    1.50,
    2.10,
    1.80,
    2.00,
    2.20,
    1.92,
    2.10,
    2.05,
    2.10,
    2.00,
    2.05,
    2.10,
    1.80
], dtype=np.float32)


def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi)
    prob = logprob.exp().sum(1)
    return prob


class ScreenScoreModel:

    def __init__(self, device='cpu'):
        self.device = device

        pt_dir = os.path.join(CURDIR, 'saved_models')
        state_dict_path = os.path.join(pt_dir, CURMODEL)

        assert os.path.exists(state_dict_path)

        try:
            self.load_model(state_dict_path, self.device)
        except Exception:

            torch.save(torch.load(state_dict_path, map_location='cpu'), state_dict_path)
            self.load_model(state_dict_path, self.device)

    def load_state_dict(self, state_dict_path):
        state_dict = torch.load(state_dict_path, map_location='cpu')
        return state_dict

    def load_model(self, state_dict_path, device):
        state_dict = self.load_state_dict(state_dict_path)['model_state_dict']

        args = {}
        args["batch_size"] = 128
        args["aux_weight"] = 0.001
        args["dist_threhold"] = 5
        args['device'] = device
        args['seeds'] = 126
        args["num_workers"] = 10
        args["cutoff"] = 10.0
        args["num_node_featsp"] = 51
        args["num_node_featsl"] = 38
        args["num_edge_featsp"] = 5
        args["num_edge_featsl"] = 6
        args["hidden_dim0"] = 256
        args["hidden_dim"] = 256
        args["n_gaussians"] = 10
        args["dropout_rate"] = 0.15
        args["outprefix"] = "mdn_v2_graph_largeob"

        ligmodel = DGLGraphTransformer(
            in_channels=101,
            edge_features=10,
            num_hidden_channels=args["hidden_dim0"],
            activ_fn=torch.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.15,
            num_layers=6
        )

        protmodel = DGLGraphTransformer(
            in_channels=48,
            edge_features=5,
            num_hidden_channels=args["hidden_dim0"],
            activ_fn=torch.nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.15,
            num_layers=6
        )

        self.model = MDNV2(
            ligmodel, protmodel,
            in_channels=args["hidden_dim0"],
            hidden_dim=args["hidden_dim"],
            n_gaussians=args["n_gaussians"],
            dropout_rate=args["dropout_rate"],
            dist_threhold=args["dist_threhold"]
        ).to(args['device'])

        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        del state_dict
        return self.model

    def freeze_param(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def __call__(self, data):
        return self.model(data)


class GridCache:

    _default_cache_path = os.path.join(CURDIR, 'grid_cache.pt')

    def __init__(self, device='cpu'):
        self.grid_cache_scale = -0.15
        self.device = device
        self.grid_spacing = DISTANCE_SPACING
        global LOADED_MODEL
        if LOADED_MODEL is None:
            self.model = ScreenScoreModel(device)
            self.model.freeze_param()
            LOADED_MODEL = self.model
        else:
            self.model = LOADED_MODEL
        self.spacing_grade = GridCache.get_spacing_grade()

    def calc_protein_embedding(self, protein_path):

        prot = from_pdb_path(protein_path)
        prot_torsions_feats = atom37_to_torsion_angles(
            prot.aatype[np.newaxis, :],
            prot.atom_positions[np.newaxis, :],
            prot.atom_mask[np.newaxis, :]
        )

        prot_torsion_sin_cos = np.array(prot_torsions_feats['torsion_angles_sin_cos'])
        torsions = np.angle(prot_torsion_sin_cos[:, :, :, 1] + 1j*prot_torsion_sin_cos[:, :, :, 0]).reshape(-1, 7)

        self.graph_protein = prot_to_graph(
            protein_path, 10,
            prot_torsions_sin_cos=torsions,
            rigidgroups_gt_frames=None,
            protein=prot
        )
        self.graph_protein = self.graph_protein.to(self.device)

        residues_positions = self.graph_protein.ndata["pos"][:, :14, :]
        self.protein_non_nan = torch.logical_not(
            torch.isnan(residues_positions.squeeze(0).sum(-1)))

        self.protein_atom_type = torch.argmax(
            self.graph_protein.ndata["atom_type"][:, :self.model.model.max_num_atoms, :],
            dim=2)
        self.protein_atom_type = self.protein_atom_type[self.protein_non_nan]

        self.protein_positions = residues_positions[self.protein_non_nan].to(self.device)

        with torch.no_grad():
            self.protein_embedding = self.model.model.forward_protein(self.graph_protein)
        return self.protein_embedding

    def calc_ligand_embedding(self, ligand_mol):
        self.graph_ligand = mol_to_graph(ligand_mol, explicit_H=False, use_chirality=True)
        self.graph_ligand = self.graph_ligand.to(self.device)
        self.ligand_positions = self.graph_ligand.ndata["pos"]
        with torch.no_grad():
            self.ligand_embedding = self.model.model.forward_ligand(self.graph_ligand)

    def calc_pi_sigma_mu(self):

        with torch.no_grad():
            self.pi, self.sigma, self.mu = self.model.model.forward_mlp(
                *self.ligand_embedding, *self.protein_embedding)

        num_ligand_atoms = len(self.graph_ligand.ndata['atom'])
        non_nan_lp = self.protein_non_nan.unsqueeze(0).tile(num_ligand_atoms, 1, 1)
        non_nan_lp = non_nan_lp.reshape([-1, self.model.model.max_num_atoms])

        idx = torch.where(non_nan_lp)
        self.pi = self.pi.permute(0, 2, 1)[idx].reshape([-1, self.model.model.n_gaussians])
        self.sigma = self.sigma.permute(0, 2, 1)[idx].reshape([-1, self.model.model.n_gaussians])
        self.mu = self.mu.permute(0, 2, 1)[idx].reshape([-1, self.model.model.n_gaussians])

    def calc_ligand_cache(self):
        num_ligand_atoms = len(self.graph_ligand.ndata['atom'])


        ligand_type = torch.argmax(self.graph_ligand.ndata['atom'][:, :17], dim=1)
        ligand_positions = self.graph_ligand.ndata["pos"]


        result_cache = []
        for grade in self.spacing_grade:
            distance = torch.full([self.pi.shape[0], 1], grade,
                                dtype=torch.float32, device=self.device)
            prob = calculate_probablity(self.pi, self.sigma, self.mu, distance)

            result_cache.append(prob.unsqueeze(1))
        result_cache = torch.cat(result_cache, dim=1).reshape(
            [num_ligand_atoms, -1, len(self.spacing_grade)])
        result_cache = result_cache.detach()

        result_cache = result_cache * self.grid_cache_scale
        return (result_cache, ligand_type, ligand_positions)

    @staticmethod
    def get_spacing_grade(distance_spacing=None, LP_threshold=None):
        if distance_spacing is None:
            distance_spacing = DISTANCE_SPACING

        if LP_threshold is None:
            LP_threshold = PL_DISTANCE_THRESHOLD

        num_spacing_grade = math.ceil(LP_threshold / distance_spacing) + 1
        spacing_grade = [distance_spacing * i for i in range(num_spacing_grade)]
        min_dis = 1e-2
        return [min_dis if i < min_dis else i for i in spacing_grade]

    @classmethod
    def cache_screen_score(cls):
        spacing_grade = cls.get_spacing_grade()
        repulsive_energy = np.zeros([len(atom_vdw), len(atom_vdw), len(spacing_grade)], dtype=np.float32)
        repulsive_grad = np.zeros_like(repulsive_energy)
        for i, a1_vdw in enumerate(atom_vdw):
            for j, a2_vdw in enumerate(atom_vdw):
                a1a2_vdw = a1_vdw + a2_vdw
                for k, dis in enumerate(spacing_grade):
                    tmp = dis - a1a2_vdw
                    if tmp > 0:
                        continue
                    e = (tmp ** 2) * 0.10
                    grad = 2 * tmp * 0.10

                    repulsive_energy[i, j, k] = e
                    repulsive_grad[i, j, k] = grad / dis
        return repulsive_energy, repulsive_grad

    @classmethod
    def generate_cache(cls, cache_path=None) -> None:
        cache_path = (cache_path or cls._default_cache_path)
        data_dict = dict(
            zip(('repulsive_energy', 'repulsive_grad'),
                cls.cache_screen_score()))
        data_dict['model'] = CURMODEL
        torch.save(data_dict, cache_path)
        return data_dict

    @classmethod
    def get_cache(cls, cache_path=None):
        global LOADED_GRID_CACHE
        if LOADED_GRID_CACHE is not None:
            return LOADED_GRID_CACHE
        cache_path = (cache_path or cls._default_cache_path)
        if not os.path.exists(cache_path):
            raise FileNotFoundError('MDNV2 grid cache file {} does not exit.'.format(cache_path))
        data_dict = torch.load(cache_path)
        if data_dict['model'] != CURMODEL:
            raise ValueError('MDNV2模型{}与缓存gridcahce.pt不一致'.format(CURMODEL))
        LOADED_GRID_CACHE = data_dict
        return LOADED_GRID_CACHE

    @classmethod
    def verify_and_generate_cache(cls, cache_path=None):
        cache_path = (cache_path or cls._default_cache_path)
        if os.path.exists(cache_path):
            data_dict = torch.load(cache_path)
            if data_dict.get('model') != CURMODEL or len(data_dict) != 6:
                os.remove(cache_path)
        if not os.path.exists(cache_path):
            cls.generate_cache()

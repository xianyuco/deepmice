import math
import os
from collections import OrderedDict

import numpy as np
import torch

from .score_model import GINNet, CURMODEL
from .utils import xscore_model_vdw, ob_model_vdw


PL_DISTANCE_THRESHOLD = 8
DISTANCE_SPACING = 0.375
MAX_SPACING_GRADE_INDEX = 16
CURDIR = os.path.dirname(os.path.abspath(__file__))

LOADED_GRID_CACHE = None


class ScreenScoreModel:

    def __init__(self, device=None):
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda:0'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        pt_dir = os.path.join(CURDIR, 'saved_models')
        state_dict_path = os.path.join(pt_dir, CURMODEL)
        assert (os.path.exists(state_dict_path))

        torch.save(torch.load(state_dict_path, map_location='cpu'), state_dict_path)
        self.load_model(state_dict_path, self.device)

    def load_state_dict(self, state_dict_path):
        state_dict = torch.load(state_dict_path, map_location='cpu')
        if next(iter(state_dict))[:7] == 'module.':
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
        return state_dict

    def load_model(self, state_dict_path, device):
        state_dict = self.load_state_dict(state_dict_path)
        self.model = GINNet()
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()
        del state_dict
        return self.model

    def freeze_param(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def forward_pair(self, a_atoms_type, a_neighbors_type, a_vdw, b_atoms_type, b_neighbors_type, b_vdw, pair_distance):
        return self.model.forward_pair(
            a_atoms_type.to(self.device),
            a_neighbors_type.to(self.device),
            a_vdw.to(self.device),
            b_atoms_type.to(self.device),
            b_neighbors_type.to(self.device),
            b_vdw.to(self.device),
            pair_distance.to(self.device))


class GridCache:

    _default_cache_path = os.path.join(CURDIR, 'grid_cache.pt')

    def __init__(self):
        self.grid_spacing = DISTANCE_SPACING

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
    def cache_screen_score(cls,
                           distance_spacing=None,
                           LP_threshold=None,
                           batch_size=10000,
                           device=None):
        spacing_grade = cls.get_spacing_grade(distance_spacing, LP_threshold)

        ModelAtomTypeSize = 26
        AtomTypeSize = ModelAtomTypeSize + 3
        AtomNeighborsSize = 16

        model = ScreenScoreModel(device)
        model.freeze_param()

        feat_shape = [AtomTypeSize, AtomNeighborsSize,
                      AtomTypeSize, AtomNeighborsSize,
                      len(spacing_grade)]

        a_atoms_type = np.zeros(feat_shape, dtype=np.int64)
        a_neighbors_type = np.zeros(feat_shape, dtype=np.int64)
        a_vdws = np.zeros(feat_shape, dtype=np.float32)
        b_atoms_type = np.zeros(feat_shape, dtype=np.int64)
        b_neighbors_type = np.zeros(feat_shape, dtype=np.int64)
        b_vdws = np.zeros(feat_shape, dtype=np.float32)
        spacing_grade_type = np.zeros(feat_shape, dtype=np.float32)

        for atype in range(AtomTypeSize):
            if atype >= ModelAtomTypeSize:
                if atype == 28:
                    afill = 25
                else:
                    afill = 3
            else:
                afill = atype
            a_atoms_type[atype] = afill
            a_vdws[atype] = ob_model_vdw[atype]
            for anei in range(AtomNeighborsSize):
                a_neighbors_type[atype, anei] = anei
                for btype in range(AtomTypeSize):
                    if btype >= ModelAtomTypeSize:
                        if atype == 28:
                            afill = 25
                        else:
                            afill = 3
                    else:
                        bfill = btype
                    b_atoms_type[atype, anei, btype] = bfill
                    b_vdws[atype, anei, btype] = ob_model_vdw[btype]
                    for bnei in range(AtomNeighborsSize):
                        b_neighbors_type[atype, anei, btype, bnei] = bnei
                        spacing_grade_type[atype, anei, btype, bnei] = spacing_grade

        a_atoms_type = torch.from_numpy(a_atoms_type).reshape(-1)
        a_neighbors_type = torch.from_numpy(a_neighbors_type).reshape(-1)
        a_vdws = torch.from_numpy(a_vdws).reshape(-1)
        b_atoms_type = torch.from_numpy(b_atoms_type).reshape(-1)
        b_neighbors_type = torch.from_numpy(b_neighbors_type).reshape(-1)
        b_vdws = torch.from_numpy(b_vdws).reshape(-1)
        spacing_grade_type = torch.from_numpy(spacing_grade_type).reshape(-1)
        spacing_grade_type.requires_grad = True
        num_pairs = len(a_atoms_type)
        loops = math.ceil(num_pairs / batch_size)
        cache_result = []
        cache_grad_result = []
        for i in range(loops):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            res = model.forward_pair(
                a_atoms_type[start_index: end_index],
                a_neighbors_type[start_index: end_index],
                a_vdws[start_index: end_index],
                b_atoms_type[start_index: end_index],
                b_neighbors_type[start_index: end_index],
                b_vdws[start_index: end_index],
                spacing_grade_type[start_index: end_index])
            cache_result.append(res.detach().data.clone())
            aux_loss = torch.sum(res)
            aux_loss.backward()
            tmp_grad = spacing_grade_type.grad.data[start_index: end_index].detach().data.clone()
            tmp_grad /= spacing_grade_type.data[start_index: end_index]
            cache_grad_result.append(tmp_grad)
        cache_result = torch.cat(cache_result, 0)
        cache_grad_result = torch.cat(cache_grad_result, 0)

        cache_result = cache_result.reshape(feat_shape)
        cache_grad_result = cache_grad_result.reshape(feat_shape)
        del model

        cache_result = -cache_result.to(torch.float32).detach().cpu().numpy()
        model_cache = cache_result.copy()
        cache_grad_result = -cache_grad_result.to(torch.float32).detach().cpu().numpy()

        vdw = np.array(xscore_model_vdw, dtype=np.float32) * 1.0
        repulsive_energy = np.zeros([AtomTypeSize, AtomTypeSize, len(spacing_grade)], dtype=np.float32)
        repulsive_grad = np.zeros_like(repulsive_energy)
        for atype in range(AtomTypeSize):
            for btype in range(AtomTypeSize):
                for idx, distance in enumerate(spacing_grade):
                    vdw_dis = vdw[atype] + vdw[btype]
                    tmp = distance - vdw_dis
                    tmp_energy_1 = -0.02 * math.exp(-((tmp / 0.5) ** 2))
                    tmp_grad_1 = tmp_energy_1 * (-8) * tmp

                    tmp_energy_2 = -0.0008 * math.exp(-(((tmp - 3) / 2) ** 2))
                    tmp_grad_2 = tmp_energy_2 * (-0.5) * (tmp - 3)

                    tmp_energy = tmp_energy_1 + tmp_energy_2
                    tmp_grad = tmp_grad_1 + tmp_grad_2
                    if tmp < 0:
                        repulsion = (tmp ** 2) * 0.21
                        repulsion_grad = 2 * tmp * 0.21
                        tmp_energy += repulsion
                        tmp_grad += repulsion_grad
                        if tmp < -1.0:
                            tmp_energy += repulsion * (-(tmp + 1.0) / 0.375)
                            tmp_grad += repulsion_grad * (-(tmp + 1.0) / 0.375)

                    if distance > 1e-2:
                        tmp_grad /= distance
                    cache_result[atype, :, btype, :, idx] += tmp_energy
                    cache_grad_result[atype, :, btype, :, idx] += tmp_grad
                    repulsive_energy[atype, btype, idx] = tmp_energy
                    repulsive_grad[atype, btype, idx] = tmp_grad

        return model_cache, cache_result, cache_grad_result, repulsive_energy, repulsive_grad

    @classmethod
    def generate_cache(cls, cache_path=None, device=None) -> None:
        cache_path = (cache_path or cls._default_cache_path)
        data_dict = dict(zip(
            ('mlp_model_cache', 'score', 'grad', 'repulsive_energy', 'repulsive_grad'),
            cls.cache_screen_score(device=device)))
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
            raise FileNotFoundError('MLP grid cache file {} does not exit.'.format(cache_path))
        data_dict = torch.load(cache_path)
        if data_dict['model'] != CURMODEL:
            raise ValueError('MLP模型{}与缓存gridcahce.pt不一致'.format(CURMODEL))
        LOADED_GRID_CACHE = data_dict
        return LOADED_GRID_CACHE

    @classmethod
    def verify_and_generate_cache(cls, cache_path=None, device=None):
        cache_path = (cache_path or cls._default_cache_path)
        if os.path.exists(cache_path):
            data_dict = torch.load(cache_path)
            if data_dict.get('model') != CURMODEL or len(data_dict) != 6:
                os.remove(cache_path)
        if not os.path.exists(cache_path):
            cls.generate_cache(device=device)

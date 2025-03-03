import torch
from torch.nn import Sequential, Linear, ReLU


CURMODEL = 'model7_only8A_plus5_all_bias_24.pt'


class GINNet(torch.nn.Module):
    def __init__(self):
        super(GINNet, self).__init__()
        self.dim = 21
        self.x_num_features = self.dim * 2 + 1
        self.embedding1 = torch.nn.Embedding(26, 13)
        self.embedding_neighbors = torch.nn.Embedding(16, 8)

        self.linear = Sequential(Linear(self.x_num_features, 128),
                                 ReLU(),
                                 Linear(128, 256),
                                 ReLU(),
                                 Linear(256, 128),
                                 ReLU(),
                                 Linear(128, 1))
        self.out_linear = Linear(3, 1, bias=True)

    def forward_pair(self, a_atoms_type, a_neighbors_type, a_vdw, b_atoms_type, b_neighbors_type, b_vdw, pair_distance):
        batch_size = a_atoms_type.shape[0]
        a_atoms_type = self.embedding1(a_atoms_type)
        b_atoms_type = self.embedding1(b_atoms_type)
        a_neighbors_type = self.embedding_neighbors(a_neighbors_type)
        b_neighbors_type = self.embedding_neighbors(b_neighbors_type)

        tmp_distance = pair_distance - a_vdw - b_vdw
        gauss = torch.exp(-1 * (tmp_distance * 2) ** 2)
        gauss = (gauss * 2).unsqueeze(-1)
        repulsion = (tmp_distance < 0).to(torch.float32) * (tmp_distance ** 2)
        repulsion = (repulsion * 2).unsqueeze(-1)

        pair_distance = pair_distance / 10.0
        pair_distance = pair_distance.unsqueeze(-1)

        xe = torch.cat(
            (a_atoms_type, a_neighbors_type, b_atoms_type, b_neighbors_type, pair_distance,
             b_atoms_type, b_neighbors_type, a_atoms_type, a_neighbors_type, pair_distance), 1)
        xe = xe.reshape([batch_size * 2, self.x_num_features])
        xe = self.linear(xe)
        xe = xe.reshape(batch_size, 2)
        xe = xe.sum(1, keepdim=True)

        out = torch.cat((xe, gauss, repulsion), 1)
        out = self.out_linear(out).squeeze() - 0.9800

        return out

import torch as th
import torch.nn.functional as F
import numpy as np
import dgl.function as fn
from torch import nn

# This file is modified from https://github.com/sc8668/RTMScore
# MIT License Copyright (c) 2023 sc8668

# the model architecture of graph transformer is modified from
# https://github.com/BioinfoMachineLearning/DeepInteract


def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        th.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, num_input_feats, num_output_feats,
                 num_heads, using_bias=False, update_edge_feats=True):
        super(MultiHeadAttentionLayer, self).__init__()

        self.num_output_feats = num_output_feats
        self.num_heads = num_heads
        self.using_bias = using_bias
        self.update_edge_feats = update_edge_feats

        self.Q = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.K = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.V = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)
        self.edge_feats_projection = nn.Linear(
            num_input_feats, self.num_output_feats * self.num_heads, bias=using_bias)

        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        scale = 2.0
        if self.using_bias:
            glorot_orthogonal(self.Q.weight, scale=scale)
            self.Q.bias.data.fill_(0)

            glorot_orthogonal(self.K.weight, scale=scale)
            self.K.bias.data.fill_(0)

            glorot_orthogonal(self.V.weight, scale=scale)
            self.V.bias.data.fill_(0)

            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)
            self.edge_feats_projection.bias.data.fill_(0)
        else:
            glorot_orthogonal(self.Q.weight, scale=scale)
            glorot_orthogonal(self.K.weight, scale=scale)
            glorot_orthogonal(self.V.weight, scale=scale)
            glorot_orthogonal(self.edge_feats_projection.weight, scale=scale)

    def propagate_attention(self, g):

        g.apply_edges(
            lambda edges: {"score": edges.src['K_h'] * edges.dst['Q_h']})

        g.apply_edges(lambda edges: {"score": (
            edges.data["score"]/np.sqrt(self.num_output_feats)).clamp(-5.0, 5.0)})

        g.apply_edges(
            lambda edges: {"score": edges.data['score'] * edges.data['proj_e']})

        if self.update_edge_feats:
            g.apply_edges(lambda edges: {"e_out": edges.data["score"]})

        g.apply_edges(lambda edges: {"score": th.exp(
            (edges.data["score"].sum(-1, keepdim=True)).clamp(-5.0, 5.0))})

        g.update_all(fn.u_mul_e('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.update_all(fn.copy_e('score', 'score'), fn.sum('score', 'z'))

    def forward(self, g, node_feats, edge_feats):
        with g.local_scope():
            e_out = None
            node_feats_q = self.Q(node_feats)
            node_feats_k = self.K(node_feats)
            node_feats_v = self.V(node_feats)
            edge_feats_projection = self.edge_feats_projection(edge_feats)

            g.ndata['Q_h'] = node_feats_q.view(-1,
                                               self.num_heads, self.num_output_feats)
            g.ndata['K_h'] = node_feats_k.view(-1,
                                               self.num_heads, self.num_output_feats)
            g.ndata['V_h'] = node_feats_v.view(-1,
                                               self.num_heads, self.num_output_feats)
            g.edata['proj_e'] = edge_feats_projection.view(
                -1, self.num_heads, self.num_output_feats)

            self.propagate_attention(g)

            h_out = g.ndata['wV'] / \
                (g.ndata['z'] + th.full_like(g.ndata['z'], 1e-6))
            if self.update_edge_feats:
                e_out = g.edata['e_out']

        return h_out, e_out


class GraphTransformerModule(nn.Module):
    def __init__(
            self,
            num_hidden_channels,
            activ_fn=nn.SiLU(),
            residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
    ):
        super(GraphTransformerModule, self).__init__()

        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers

        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,

            self.num_hidden_channels != self.num_output_feats,
            update_edge_feats=True
        )

        self.O_node_feats = nn.Linear(
            self.num_output_feats, self.num_output_feats)
        self.O_edge_feats = nn.Linear(
            self.num_output_feats, self.num_output_feats)

        dropout = nn.Dropout(
            p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats,
                      self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2,
                      self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm2_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm2_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.edge_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats,
                      self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2,
                      self.num_output_feats, bias=False)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)
        glorot_orthogonal(self.O_edge_feats.weight, scale=scale)
        self.O_edge_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

        for layer in self.edge_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, g, node_feats, edge_feats):
        """Perform a forward pass of graph attention using a multi-head attention (MHA) module."""
        node_feats_in1 = node_feats

        edge_feats_in1 = edge_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        node_attn_out, edge_attn_out = self.mha_module(
            g, node_feats, edge_feats)

        node_feats = node_attn_out.view(-1, self.num_output_feats)
        edge_feats = edge_attn_out.view(-1, self.num_output_feats)

        node_feats = F.dropout(
            node_feats, self.dropout_rate, training=self.training)
        edge_feats = F.dropout(
            edge_feats, self.dropout_rate, training=self.training)

        node_feats = self.O_node_feats(node_feats)
        edge_feats = self.O_edge_feats(edge_feats)

        if self.residual:
            node_feats = node_feats_in1 + node_feats
            edge_feats = edge_feats_in1 + edge_feats

        node_feats_in2 = node_feats

        edge_feats_in2 = edge_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
            edge_feats = self.layer_norm2_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm2_node_feats(node_feats)
            edge_feats = self.batch_norm2_edge_feats(edge_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)
        for layer in self.edge_feats_MLP:
            edge_feats = layer(edge_feats)

        if self.residual:
            node_feats = node_feats_in2 + node_feats
            edge_feats = edge_feats_in2 + edge_feats

        return node_feats, edge_feats

    def forward(self, g, node_feats, edge_feats):
        node_feats, edge_feats = self.run_gt_layer(g, node_feats, edge_feats)
        return node_feats, edge_feats


class FinalGraphTransformerModule(nn.Module):
    def __init__(self,
                 num_hidden_channels,
                 activ_fn=nn.SiLU(),
                 residual=True,
                 num_attention_heads=4,
                 norm_to_apply='batch',
                 dropout_rate=0.1,
                 num_layers=4):
        super(FinalGraphTransformerModule, self).__init__()

        self.activ_fn = activ_fn
        self.residual = residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.apply_layer_norm = 'layer' in self.norm_to_apply.lower()

        self.num_hidden_channels, self.num_output_feats = num_hidden_channels, num_hidden_channels
        if self.apply_layer_norm:
            self.layer_norm1_node_feats = nn.LayerNorm(self.num_output_feats)
            self.layer_norm1_edge_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm1_node_feats = nn.BatchNorm1d(self.num_output_feats)
            self.batch_norm1_edge_feats = nn.BatchNorm1d(self.num_output_feats)

        self.mha_module = MultiHeadAttentionLayer(
            self.num_hidden_channels,
            self.num_output_feats // self.num_attention_heads,
            self.num_attention_heads,
            self.num_hidden_channels != self.num_output_feats,
            update_edge_feats=False)

        self.O_node_feats = nn.Linear(
            self.num_output_feats, self.num_output_feats)

        dropout = nn.Dropout(
            p=self.dropout_rate) if self.dropout_rate > 0.0 else nn.Identity()
        self.node_feats_MLP = nn.ModuleList([
            nn.Linear(self.num_output_feats,
                      self.num_output_feats * 2, bias=False),
            self.activ_fn,
            dropout,
            nn.Linear(self.num_output_feats * 2,
                      self.num_output_feats, bias=False)
        ])

        if self.apply_layer_norm:
            self.layer_norm2_node_feats = nn.LayerNorm(self.num_output_feats)
        else:
            self.batch_norm2_node_feats = nn.BatchNorm1d(self.num_output_feats)

        self.reset_parameters()

    def reset_parameters(self):
        scale = 2.0
        glorot_orthogonal(self.O_node_feats.weight, scale=scale)
        self.O_node_feats.bias.data.fill_(0)

        for layer in self.node_feats_MLP:
            if hasattr(layer, 'weight'):
                glorot_orthogonal(layer.weight, scale=scale)

    def run_gt_layer(self, g, node_feats, edge_feats):
        node_feats_in1 = node_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm1_node_feats(node_feats)
            edge_feats = self.layer_norm1_edge_feats(edge_feats)
        else:
            node_feats = self.batch_norm1_node_feats(node_feats)
            edge_feats = self.batch_norm1_edge_feats(edge_feats)

        node_attn_out, _ = self.mha_module(g, node_feats, edge_feats)
        node_feats = node_attn_out.view(-1, self.num_output_feats)
        node_feats = F.dropout(
            node_feats, self.dropout_rate, training=self.training)
        node_feats = self.O_node_feats(node_feats)

        if self.residual:
            node_feats = node_feats_in1 + node_feats

        node_feats_in2 = node_feats

        if self.apply_layer_norm:
            node_feats = self.layer_norm2_node_feats(node_feats)
        else:
            node_feats = self.batch_norm2_node_feats(node_feats)

        for layer in self.node_feats_MLP:
            node_feats = layer(node_feats)

        if self.residual:
            node_feats = node_feats_in2 + node_feats
        return node_feats

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.run_gt_layer(g, node_feats, edge_feats)
        return node_feats


class DGLGraphTransformer(nn.Module):
    def __init__(
            self,
            in_channels,
            edge_features=10,
            num_hidden_channels=128,
            activ_fn=nn.SiLU(),
            transformer_residual=True,
            num_attention_heads=4,
            norm_to_apply='batch',
            dropout_rate=0.1,
            num_layers=4,
            **kwargs
    ):
        super(DGLGraphTransformer, self).__init__()

        self.activ_fn = activ_fn
        self.transformer_residual = transformer_residual
        self.num_attention_heads = num_attention_heads
        self.norm_to_apply = norm_to_apply
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.node_encoder = nn.Linear(in_channels, num_hidden_channels)
        self.edge_encoder = nn.Linear(edge_features, num_hidden_channels)

        num_intermediate_layers = max(0, num_layers - 1)
        gt_block_modules = [GraphTransformerModule(
            num_hidden_channels=num_hidden_channels,
            activ_fn=activ_fn,
            residual=transformer_residual,
            num_attention_heads=num_attention_heads,
            norm_to_apply=norm_to_apply,
            dropout_rate=dropout_rate,
            num_layers=num_layers) for _ in range(num_intermediate_layers)]
        if num_layers > 0:
            gt_block_modules.extend([
                FinalGraphTransformerModule(
                    num_hidden_channels=num_hidden_channels,
                    activ_fn=activ_fn,
                    residual=transformer_residual,
                    num_attention_heads=num_attention_heads,
                    norm_to_apply=norm_to_apply,
                    dropout_rate=dropout_rate,
                    num_layers=num_layers)])
        self.gt_block = nn.ModuleList(gt_block_modules)

    def forward(self, g, node_feats, edge_feats):
        node_feats = self.node_encoder(node_feats)
        edge_feats = self.edge_encoder(edge_feats)

        g.ndata['x'] = node_feats
        g.edata['h'] = edge_feats

        for gt_layer in self.gt_block[:-1]:
            node_feats, edge_feats = gt_layer(g, node_feats, edge_feats)
        node_feats = self.gt_block[-1](g, node_feats, edge_feats)
        return node_feats


def to_dense_batch_dgl(bg, feats, fill_value=0):
    max_num_nodes = int(bg.batch_num_nodes().max())
    batch = th.cat([th.full((1, x.type(th.int)), y) for x, y in zip(bg.batch_num_nodes(
    ), range(bg.batch_size))], dim=1).reshape(-1).type(th.long).to(bg.device)
    cum_nodes = th.cat(
        [batch.new_zeros(1), bg.batch_num_nodes().cumsum(dim=0)])
    idx = th.arange(bg.num_nodes(), dtype=th.long, device=bg.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    size = [bg.batch_size * max_num_nodes] + list(feats.size())[1:]
    out = feats.new_full(size, fill_value)
    out[idx] = feats
    out = out.view([bg.batch_size, max_num_nodes] + list(feats.size())[1:])
    mask = th.zeros(bg.batch_size * max_num_nodes, dtype=th.bool,
                    device=bg.device)
    mask[idx] = 1
    mask = mask.view(bg.batch_size, max_num_nodes)
    return out, mask


class MDNV2(nn.Module):
    def __init__(self, lig_model, prot_model, in_channels, hidden_dim, n_gaussians, dropout_rate=0.15,
                 dist_threhold=1000):
        super(MDNV2, self).__init__()
        self.max_num_atoms = 14
        self.lig_model = lig_model
        self.prot_model = prot_model
        self.MLP = nn.Sequential(nn.Linear(in_channels*2, hidden_dim),
                                 nn.BatchNorm1d(hidden_dim),
                                 nn.ELU(),
                                 nn.Dropout(p=dropout_rate)
                                 )
        self.z_pi = nn.Linear(hidden_dim, n_gaussians*self.max_num_atoms)
        self.z_sigma = nn.Linear(hidden_dim, n_gaussians*self.max_num_atoms)
        self.z_mu = nn.Linear(hidden_dim, n_gaussians*self.max_num_atoms)
        self.atom_types = nn.Linear(in_channels, 17)
        self.bond_types = nn.Linear(in_channels*2, 5)
        self.n_gaussians = n_gaussians
        self.dist_threhold = dist_threhold

    def forward(self, bgp, bgl):
        h_l = self.lig_model(
            bgl, bgl.ndata['atom'].float(), bgl.edata['bond'].float())
        h_p = self.prot_model(
            bgp, bgp.ndata['feats'].float(), bgp.edata['feats'].float())
        print(bgp.ndata["atom_type"].shape, '1111')

        h_l_x, l_mask = to_dense_batch_dgl(bgl, h_l, fill_value=0)
        h_p_x, p_mask = to_dense_batch_dgl(bgp, h_p, fill_value=0)

        h_l_pos, _ = to_dense_batch_dgl(bgl, bgl.ndata["pos"], fill_value=0)
        h_p_pos, _ = to_dense_batch_dgl(
            bgp, bgp.ndata["pos"][:, :self.max_num_atoms], fill_value=0)
        (B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)
        self.B = B
        self.N_l = N_l
        self.N_p = N_p
        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_p, 1)
        h_p_x = h_p_x.unsqueeze(-3)
        h_p_x = h_p_x.repeat(1, N_l, 1, 1)
        C = th.cat((h_l_x, h_p_x), -1)
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
        self.C = C = C[C_mask]
        C = self.MLP(C)
        C_batch = th.tensor(
            range(B), device=C.device).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]

        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1
        atom_types = self.atom_types(h_l)
        bond_types = self.bond_types(
            th.cat([h_l[bgl.edges()[0]], h_l[bgl.edges()[1]]], axis=1))

        dist = self.compute_euclidean_distances_matrix(
            h_l_pos, h_p_pos.view(B, -1, 3))

        dist = dist[C_mask]

        return (pi.view(-1, self.n_gaussians, self.max_num_atoms),
                sigma.view(-1, self.n_gaussians, self.max_num_atoms),
                mu.view(-1, self.n_gaussians, self.max_num_atoms),
                dist.unsqueeze(1).detach(), atom_types, bond_types, C_batch)

    def forward_pair(self, bgp, bgl):
        h_p = self.prot_model(
            bgp, bgp.ndata['feats'].float(), bgp.edata['feats'].float())
        h_p_x, p_mask = to_dense_batch_dgl(bgp, h_p, fill_value=0)

        h_l = self.lig_model(
            bgl, bgl.ndata['atom'].float(), bgl.edata['bond'].float())
        h_l_x, l_mask = to_dense_batch_dgl(bgl, h_l, fill_value=0)

        (B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)

        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_p, 1)

        h_p_x = h_p_x.unsqueeze(-3)
        h_p_x = h_p_x.repeat(1, N_l, 1, 1)

        C = th.cat((h_l_x, h_p_x), -1)
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        C_batch = th.tensor(
            range(B), device=C.device).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]

        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C))+1.1
        mu = F.elu(self.z_mu(C))+1

        return (pi.view(-1, self.n_gaussians, self.max_num_atoms),
                sigma.view(-1, self.n_gaussians, self.max_num_atoms),
                mu.view(-1, self.n_gaussians, self.max_num_atoms))

    def forward_protein(self, bgp):
        h_p = self.prot_model(
            bgp, bgp.ndata['feats'].float(), bgp.edata['feats'].float())
        h_p_x, p_mask = to_dense_batch_dgl(bgp, h_p, fill_value=0)
        return h_p_x, p_mask

    def forward_ligand(self, bgl):
        h_l = self.lig_model(
            bgl, bgl.ndata['atom'].float(), bgl.edata['bond'].float())
        h_l_x, l_mask = to_dense_batch_dgl(bgl, h_l, fill_value=0)
        return h_l_x, l_mask

    def forward_mlp(self, h_l_x, l_mask, h_p_x, p_mask):
        (B, N_l, C_out), N_p = h_l_x.size(), h_p_x.size(1)

        h_l_x = h_l_x.unsqueeze(-2)
        h_l_x = h_l_x.repeat(1, 1, N_p, 1)

        h_p_x = h_p_x.unsqueeze(-3)
        h_p_x = h_p_x.repeat(1, N_l, 1, 1)

        C = th.cat((h_l_x, h_p_x), -1)
        self.C_mask = C_mask = l_mask.view(B, N_l, 1) & p_mask.view(B, 1, N_p)
        self.C = C = C[C_mask]
        C = self.MLP(C)

        C_batch = th.tensor(
            range(B), device=C.device).unsqueeze(-1).unsqueeze(-1)
        C_batch = C_batch.repeat(1, N_l, N_p)[C_mask]

        pi = F.softmax(self.z_pi(C), -1)
        sigma = F.elu(self.z_sigma(C)) + 1.1
        mu = F.elu(self.z_mu(C)) + 1
        return (pi.view(-1, self.n_gaussians, self.max_num_atoms),
                sigma.view(-1, self.n_gaussians, self.max_num_atoms),
                mu.view(-1, self.n_gaussians, self.max_num_atoms))

    def compute_euclidean_distances_matrix(self, X, Y):
        X = X.double()
        Y = Y.double()
        dists = -2 * th.bmm(X, Y.permute(0, 2, 1)) + th.sum(Y**2,
                                                            axis=-1).unsqueeze(1) + th.sum(X**2, axis=-1).unsqueeze(-1)
        return th.nan_to_num((dists**0.5).view(self.B, self.N_l, -1, self.max_num_atoms), 10000)

# This file is modified from https://github.com/sc8668/RTMScore
# MIT License Copyright (c) 2023 sc8668

import random

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Normal
from torch_scatter import scatter_add
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from scipy.stats import pearsonr, spearmanr
import dgl


class Meter(object):
    def __init__(self, mean=None, std=None):
        self.mask = []
        self.y_pred = []
        self.y_true = []

        if (mean is not None) and (std is not None):
            self.mean = mean.cpu()
            self.std = std.cpu()
        else:
            self.mean = None
            self.std = None

    def update(self, y_pred, y_true, mask=None):
        self.y_pred.append(y_pred.detach().cpu())
        self.y_true.append(y_true.detach().cpu())
        if mask is None:
            self.mask.append(th.ones(self.y_pred[-1].shape))
        else:
            self.mask.append(mask.detach().cpu())

    def _finalize(self):
        mask = th.cat(self.mask, dim=0)
        y_pred = th.cat(self.y_pred, dim=0)
        y_true = th.cat(self.y_true, dim=0)

        if (self.mean is not None) and (self.std is not None):
            y_pred = y_pred * self.std + self.mean

        return mask, y_pred, y_true

    def _reduce_scores(self, scores, reduction='none'):
        if reduction == 'none':
            return scores
        elif reduction == 'mean':
            return np.mean(scores)
        elif reduction == 'sum':
            return np.sum(scores)
        else:
            raise ValueError(
                "Expect reduction to be 'none', 'mean' or 'sum', got {}".format(reduction))

    def multilabel_score(self, score_func, reduction='none'):
        mask, y_pred, y_true = self._finalize()
        n_tasks = y_true.shape[1]
        scores = []
        for task in range(n_tasks):
            task_w = mask[:, task]
            task_y_true = y_true[:, task][task_w != 0]
            task_y_pred = y_pred[:, task][task_w != 0]
            task_score = score_func(task_y_true, task_y_pred)
            if task_score is not None:
                scores.append(task_score)
        return self._reduce_scores(scores, reduction)

    def pearson_r(self, reduction='none'):
        def score(y_true, y_pred):
            return pearsonr(y_true.numpy(), y_pred.numpy())[0]

        return self.multilabel_score(score, reduction)

    def spearman_r(self, reduction='none'):
        def score(y_true, y_pred):
            return spearmanr(y_true.numpy(), y_pred.numpy())[0]

        return self.multilabel_score(score, reduction)

    def mae(self, reduction='none'):
        def score(y_true, y_pred):
            return F.l1_loss(y_true, y_pred).data.item()

        return self.multilabel_score(score, reduction)

    def rmse(self, reduction='none'):
        def score(y_true, y_pred):
            return th.sqrt(F.mse_loss(y_pred, y_true).cpu()).item()

        return self.multilabel_score(score, reduction)

    def roc_auc_score(self, reduction='none'):
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'ROC AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                return roc_auc_score(y_true.long().numpy(), th.sigmoid(y_pred).numpy())

        return self.multilabel_score(score, reduction)

    def pr_auc_score(self, reduction='none'):
        assert (self.mean is None) and (self.std is None), \
            'Label normalization should not be performed for binary classification.'

        def score(y_true, y_pred):
            if len(y_true.unique()) == 1:
                print('Warning: Only one class {} present in y_true for a task. '
                      'PR AUC score is not defined in that case.'.format(y_true[0]))
                return None
            else:
                precision, recall, _ = precision_recall_curve(
                    y_true.long().numpy(), th.sigmoid(y_pred).numpy())
                return auc(recall, precision)

        return self.multilabel_score(score, reduction)

    def compute_metric(self, metric_name, reduction='none'):
        if metric_name == 'rp':
            return self.pearson_r(reduction)
        elif metric_name == 'rs':
            return self.spearman_r(reduction)
        elif metric_name == 'mae':
            return self.mae(reduction)
        elif metric_name == 'rmse':
            return self.rmse(reduction)
        elif metric_name == 'roc_auc_score':
            return self.roc_auc_score(reduction)
        elif metric_name == 'pr_auc_score':
            return self.pr_auc_score(reduction)
        elif metric_name == 'return_pred_true':
            return self.return_pred_true()
        else:
            raise ValueError('Expect metric_name to be "rp" or "rs" or "mae" or "rmse" '
                             'or "roc_auc_score" or "pr_auc", got {}'.format(metric_name))


class EarlyStopping(object):
    def __init__(self, mode='higher', patience=10, filename=None, metric=None):
        if filename is None:

            filename = 'early_stop.pth'

        if metric is not None:
            assert metric in ['rp', 'rs', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score'], \
                "Expect metric to be 'rp' or 'rs' or 'mae' or " \
                "'rmse' or 'roc_auc_score', got {}".format(metric)
            if metric in ['rp', 'rs', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.timestep = 0
        self.filename = filename
        self.best_score = None
        self.early_stop = False

    def _check_higher(self, score, prev_best_score):
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        return score < prev_best_score

    def step(self, score, model):
        self.timestep += 1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        else:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop

    def save_checkpoint(self, model):
        th.save({'model_state_dict': model.state_dict(),
                 'timestep': self.timestep}, self.filename)

    def load_checkpoint(self, model):
        model.load_state_dict(th.load(self.filename)['model_state_dict'])


def mdn_loss_fn(pi, sigma, mu, y, eps=1e-10):
    normal = Normal(mu, sigma)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -th.logsumexp(th.log(pi + eps) + loglik, dim=1)

    return loss


def run_a_train_epoch(epoch, model, data_loader, optimizer,
                      enable_amp=False, scaler=None, max_node=1500,
                      update_freq=1, aux_weight=0.001, device='cpu'):
    model.train()
    total_loss = 0
    mdn_loss = 0
    atom_loss = 0
    bond_loss = 0
    for batch_id, batch_data in enumerate(data_loader):
        dist_threhold = 7
        pdbids, bgl, bgp = batch_data
        bgl = bgl.to(device)
        bgp = bgp.to(device)

        if bgp.ndata['feats'].shape[0] > max_node:
            ubgp = dgl.unbatch(bgp)
            ubgl = dgl.unbatch(bgl)
            bgps = []
            bgp = []
            bgls = []
            bgl = []
            n = 0
            for p, l in zip(ubgp, ubgl):
                if n + p.ndata['feats'].shape[0] > max_node:
                    bgps.append(dgl.batch(bgp))
                    bgls.append(dgl.batch(bgl))
                    bgp = [p]
                    bgl = [l]
                    n = p.ndata['feats'].shape[0]
                else:
                    n += p.ndata['feats'].shape[0]
                    bgp.append(p)
                    bgl.append(l)
        else:
            bgps = [bgp]
            bgls = [bgl]

        for bgp, bgl in zip(bgps, bgls):
            with th.autocast(device_type='cpu', dtype=th.float16, enabled=enable_amp):
                atom_labels = th.argmax(
                    bgl.ndata["atom"][:, :17].float(), dim=1, keepdim=False)
                bond_labels = th.argmax(
                    bgl.edata["bond"][:, :5].float(), dim=1, keepdim=False)
                pi, sigma, mu, dist, atom_types, bond_types, batch = model(
                    bgp, bgl)
                mdn = mdn_loss_fn(pi, sigma, mu, dist)

                mdn = mdn[th.where(dist.squeeze(1) <= dist_threhold)]

                mdn = mdn.mean()
                atom = F.cross_entropy(atom_types, atom_labels)
                bond = F.cross_entropy(bond_types, bond_labels)
                loss = mdn + (atom * aux_weight) + (bond * aux_weight)
                loss = loss/update_freq/len(bgps)

            scaler.scale(loss).backward()

        if ((batch_id + 1) % update_freq == 0) or (batch_id + 1 == len(data_loader)):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * bgl.batch_size
        mdn_loss += mdn.item() * bgl.batch_size
        atom_loss += atom.item() * bgl.batch_size
        bond_loss += bond.item() * bgl.batch_size

        if np.isinf(mdn_loss) or np.isnan(mdn_loss):
            break
        del bgl, bgp, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch, mdn, atom, bond, loss
        th.cuda.empty_cache()

    return total_loss / len(data_loader.dataset), mdn_loss / len(data_loader.dataset), atom_loss / len(
        data_loader.dataset), bond_loss / len(data_loader.dataset)


def run_an_eval_epoch(model, data_loader, pred=False, atom_contribution=False, res_contribution=False,
                      dist_threhold=None, aux_weight=0.001, device='cpu'):
    model.eval()
    total_loss = 0
    mdn_loss = 0
    atom_loss = 0
    bond_loss = 0
    probs = []
    at_contrs = []
    res_contrs = []
    with th.no_grad():
        for batch_id, batch_data in enumerate(data_loader):
            pdbids, bgl, bgp = batch_data
            bgl = bgl.to(device)
            bgp = bgp.to(device)
            atom_labels = th.argmax(
                bgl.ndata["atom"][:, :17], dim=1, keepdim=False)
            bond_labels = th.argmax(
                bgl.edata["bond"][:, :5], dim=1, keepdim=False)

            pi, sigma, mu, dist, atom_types, bond_types, batch = model(
                bgp, bgl)

            if pred or atom_contribution or res_contribution:
                prob = calculate_probablity(pi, sigma, mu, dist)
                if dist_threhold is not None:
                    prob[th.where(dist.squeeze(1) > dist_threhold)] = 0.
                print(prob[th.where(dist.squeeze(1) < dist_threhold)].shape)
                batch = batch.to(device)
                if pred:
                    probx = scatter_add(prob, batch, dim=0,
                                        dim_size=bgl.batch_size)

                    probs.append(probx.sum(1))
                if atom_contribution or res_contribution:
                    contribs = [prob[batch == i].reshape((bgl.batch_num_nodes()[i], bgp.batch_num_nodes()[i])) for i in
                                range(bgl.batch_size)]
                    if atom_contribution:
                        at_contrs.extend(
                            [contribs[i].sum(1).cpu().detach().numpy() for i in range(bgl.batch_size)])
                    if res_contribution:
                        res_contrs.extend(
                            [contribs[i].sum(0).cpu().detach().numpy() for i in range(bgl.batch_size)])

            else:
                tdist_threhold = 7
                mdn = mdn_loss_fn(pi, sigma, mu, dist)
                mdn = mdn[th.where(dist.squeeze(1) <= tdist_threhold)]
                mdn = mdn.mean()
                atom = F.cross_entropy(atom_types, atom_labels)
                bond = F.cross_entropy(bond_types, bond_labels)
                loss = mdn + (atom * aux_weight) + (bond * aux_weight)

                total_loss += loss.item() * bgl.batch_size
                mdn_loss += mdn.item() * bgl.batch_size
                atom_loss += atom.item() * bgl.batch_size
                bond_loss += bond.item() * bgl.batch_size

            del bgl, bgp, atom_labels, bond_labels, pi, sigma, mu, dist, atom_types, bond_types, batch
            th.cuda.empty_cache()

    if atom_contribution or res_contribution:
        if pred:
            preds = th.cat(probs)
            return [preds.cpu().detach().numpy(), at_contrs, res_contrs]
        else:
            return [None, at_contrs, res_contrs]
    else:
        if pred:
            preds = th.cat(probs)
            return preds.cpu().detach().numpy()
        else:
            return total_loss / len(data_loader.dataset), mdn_loss / len(data_loader.dataset), atom_loss / len(
                data_loader.dataset), bond_loss / len(data_loader.dataset)


def calculate_probablity(pi, sigma, mu, y):
    normal = Normal(mu, sigma)
    logprob = normal.log_prob(y.expand_as(normal.loc))

    logprob += th.log(pi)

    prob = logprob.exp().sum(1)
    print(prob.shape)
    return prob


def collate(data):
    pdbids, graphsl, graphsp = map(list, zip(*data))
    bgl = dgl.batch(graphsl)
    bgp = dgl.batch(graphsp)
    for nty in bgl.ntypes:
        bgl.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bgl.canonical_etypes:
        bgl.set_e_initializer(dgl.init.zero_initializer, etype=ety)
    for nty in bgp.ntypes:
        bgp.set_n_initializer(dgl.init.zero_initializer, ntype=nty)
    for ety in bgp.canonical_etypes:
        bgp.set_e_initializer(dgl.init.zero_initializer, etype=ety)
    return pdbids, bgl, bgp


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)

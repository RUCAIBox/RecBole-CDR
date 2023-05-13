# -*- coding: utf-8 -*-
# @Time   : 2022/3/23
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com

r"""
BiTGCF
################################################
Reference:
    Meng Liu et al. "Cross Domain Recommendation via Bi-directional Transfer Graph Collaborative Filtering Networks." in CIKM 2020.
"""

import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class BiTGCF(CrossDomainRecommender):
    r"""BiTGCF uses feature propagation and feature transfer to achieve bidirectional
        knowledge transfer between the two domains.
        We extend the basic BiTGCF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(BiTGCF, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.domain_lambda_source = config['lambda_source']  # float32 type: the weight of source embedding in transfer function
        self.domain_lambda_target = config['lambda_target']  # float32 type: the weight of target embedding in transfer function
        self.drop_rate = config['drop_rate']  # float32 type: the dropout rate
        self.connect_way = config['connect_way']  # str type: the connect way for all layers

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)

        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.dropout = nn.Dropout(p=self.drop_rate)
        self.loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.reg_loss = EmbLoss()

        # generate intermediate data
        self.source_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='source').astype(np.float32)
        self.target_interaction_matrix = dataset.inter_matrix(form='coo', value_field=None, domain='target').astype(np.float32)
        self.source_norm_adj_matrix = self.get_norm_adj_mat(self.source_interaction_matrix, self.total_num_users,
                                                       self.total_num_items).to(self.device)
        self.target_norm_adj_matrix = self.get_norm_adj_mat(self.target_interaction_matrix, self.total_num_users,
                                                       self.total_num_items).to(self.device)

        self.source_user_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=1)).to(self.device)
        self.target_user_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=1)).to(self.device)
        self.source_item_degree_count = torch.from_numpy(self.source_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)
        self.target_item_degree_count = torch.from_numpy(self.target_interaction_matrix.sum(axis=0)).transpose(0, 1).to(self.device)

        # storage variables for full sort evaluation acceleration
        self.target_restore_user_e = None
        self.target_restore_item_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ['target_restore_user_e', 'target_restore_item_e']

    def get_norm_adj_mat(self, interaction_matrix, n_users=None, n_items=None):
        # build adj matrix
        if n_users == None or n_items == None:
            n_users, n_items = interaction_matrix.shape
        A = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
        inter_M = interaction_matrix
        inter_M_t = interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self, domain='source'):
        if domain == 'source':
            user_embeddings = self.source_user_embedding.weight
            item_embeddings = self.source_item_embedding.weight
            norm_adj_matrix = self.source_norm_adj_matrix
        else:
            user_embeddings = self.target_user_embedding.weight
            item_embeddings = self.target_item_embedding.weight
            norm_adj_matrix = self.target_norm_adj_matrix
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings, norm_adj_matrix

    def graph_layer(self, adj_matrix, all_embeddings):
        side_embeddings = torch.sparse.mm(adj_matrix, all_embeddings)
        new_embeddings = side_embeddings + torch.mul(all_embeddings, side_embeddings)
        new_embeddings = all_embeddings + new_embeddings
        new_embeddings = self.dropout(new_embeddings)
        return new_embeddings

    def transfer_layer(self, source_all_embeddings, target_all_embeddings):
        source_user_embeddings, source_item_embeddings = torch.split(source_all_embeddings, [self.total_num_users, self.total_num_items])
        target_user_embeddings, target_item_embeddings = torch.split(target_all_embeddings, [self.total_num_users, self.total_num_items])

        source_user_embeddings_lam = self.domain_lambda_source * source_user_embeddings + (1 - self.domain_lambda_source) * target_user_embeddings
        target_user_embeddings_lam = self.domain_lambda_target * target_user_embeddings + (1 - self.domain_lambda_target) * source_user_embeddings
        source_item_embeddings_lam = self.domain_lambda_source * source_item_embeddings + (1 - self.domain_lambda_source) * target_item_embeddings
        target_item_embeddings_lam = self.domain_lambda_target * target_item_embeddings + (1 - self.domain_lambda_target) * source_item_embeddings

        source_user_laplace = self.source_user_degree_count
        target_user_laplace = self.target_user_degree_count
        user_laplace = source_user_laplace + target_user_laplace + 1e-7
        source_user_embeddings_lap = (source_user_laplace * source_user_embeddings + target_user_laplace * target_user_embeddings) / user_laplace
        target_user_embeddings_lap = source_user_embeddings_lap
        source_item_laplace = self.source_item_degree_count
        target_item_laplace = self.target_item_degree_count
        item_laplace = source_item_laplace + target_item_laplace + 1e-7
        source_item_embeddings_lap = (source_item_laplace * source_item_embeddings + target_item_laplace * target_item_embeddings) / item_laplace
        target_item_embeddings_lap = source_item_embeddings_lap

        source_specific_user_embeddings = source_user_embeddings[self.overlapped_num_users:]
        target_specific_user_embeddings = target_user_embeddings[self.overlapped_num_users:]
        source_specific_item_embeddings = source_item_embeddings[self.overlapped_num_items:]
        target_specific_item_embeddings = target_item_embeddings[self.overlapped_num_items:]
        source_overlap_user_embeddings = (source_user_embeddings_lam[:self.overlapped_num_users] + source_user_embeddings_lap[:self.overlapped_num_users]) / 2
        target_overlap_user_embeddings = (target_user_embeddings_lam[:self.overlapped_num_users] + target_user_embeddings_lap[:self.overlapped_num_users]) / 2
        source_overlap_item_embeddings = (source_item_embeddings_lam[:self.overlapped_num_items] + source_item_embeddings_lap[:self.overlapped_num_items]) / 2
        target_overlap_item_embeddings = (target_item_embeddings_lam[:self.overlapped_num_items] + target_item_embeddings_lap[:self.overlapped_num_items]) / 2
        source_transfer_user_embeddings = torch.cat([source_overlap_user_embeddings, source_specific_user_embeddings], dim=0)
        target_transfer_user_embeddings = torch.cat([target_overlap_user_embeddings, target_specific_user_embeddings], dim=0)
        source_transfer_item_embeddings = torch.cat([source_overlap_item_embeddings, source_specific_item_embeddings], dim=0)
        target_transfer_item_embeddings = torch.cat([target_overlap_item_embeddings, target_specific_item_embeddings], dim=0)

        source_alltransfer_embeddings = torch.cat([source_transfer_user_embeddings, source_transfer_item_embeddings], dim=0)
        target_alltransfer_embeddings = torch.cat([target_transfer_user_embeddings, target_transfer_item_embeddings], dim=0)
        return source_alltransfer_embeddings, target_alltransfer_embeddings

    def forward(self):
        source_all_embeddings, source_norm_adj_matrix = self.get_ego_embeddings(domain='source')
        target_all_embeddings, target_norm_adj_matrix = self.get_ego_embeddings(domain='target')

        source_embeddings_list = [source_all_embeddings]
        target_embeddings_list = [target_all_embeddings]
        for layer_idx in range(self.n_layers):
            source_all_embeddings = self.graph_layer(source_norm_adj_matrix, source_all_embeddings)
            target_all_embeddings = self.graph_layer(target_norm_adj_matrix, target_all_embeddings)

            source_all_embeddings, target_all_embeddings = self.transfer_layer(source_all_embeddings, target_all_embeddings)

            source_norm_embeddings = nn.functional.normalize(source_all_embeddings, p=2, dim=1)
            target_norm_embeddings = nn.functional.normalize(target_all_embeddings, p=2, dim=1)
            source_embeddings_list.append(source_norm_embeddings)
            target_embeddings_list.append(target_norm_embeddings)

        if self.connect_way == 'concat':
            source_lightgcn_all_embeddings = torch.cat(source_embeddings_list, 1)
            target_lightgcn_all_embeddings = torch.cat(target_embeddings_list, 1)
        elif self.connect_way == 'mean':
            source_lightgcn_all_embeddings = torch.stack(source_embeddings_list, dim=1)
            source_lightgcn_all_embeddings = torch.mean(source_lightgcn_all_embeddings, dim=1)
            target_lightgcn_all_embeddings = torch.stack(target_embeddings_list, dim=1)
            target_lightgcn_all_embeddings = torch.mean(target_lightgcn_all_embeddings, dim=1)

        source_user_all_embeddings, source_item_all_embeddings = torch.split(source_lightgcn_all_embeddings,
                                                                   [self.total_num_users, self.total_num_items])
        target_user_all_embeddings, target_item_all_embeddings = torch.split(target_lightgcn_all_embeddings,
                                                                   [self.total_num_users, self.total_num_items])

        return source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings

    def calculate_loss(self, interaction):
        self.init_restore_e()
        source_user_all_embeddings, source_item_all_embeddings, target_user_all_embeddings, target_item_all_embeddings = self.forward()

        losses = []

        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        source_u_embeddings = source_user_all_embeddings[source_user]
        source_i_embeddings = source_item_all_embeddings[source_item]
        target_u_embeddings = target_user_all_embeddings[target_user]
        target_i_embeddings = target_item_all_embeddings[target_item]

        # calculate BCE Loss in source domain
        source_output = self.sigmoid(torch.mul(source_u_embeddings, source_i_embeddings).sum(dim=1))
        source_bce_loss = self.loss(source_output, source_label)

        # calculate Reg Loss in source domain
        u_ego_embeddings = self.source_user_embedding(source_user)
        i_ego_embeddings = self.source_item_embedding(source_item)
        source_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)

        source_loss = source_bce_loss + self.reg_weight * source_reg_loss
        losses.append(source_loss)

        # calculate BCE Loss in target domain
        target_output = self.sigmoid(torch.mul(target_u_embeddings, target_i_embeddings).sum(dim=1))
        target_bce_loss = self.loss(target_output, target_label)

        # calculate Reg Loss in target domain
        u_ego_embeddings = self.target_user_embedding(target_user)
        i_ego_embeddings = self.target_item_embedding(target_item)
        target_reg_loss = self.reg_loss(u_ego_embeddings, i_ego_embeddings)

        target_loss = target_bce_loss + self.reg_weight * target_reg_loss
        losses.append(target_loss)

        return tuple(losses)

    def predict(self, interaction):
        result = []
        _, _, target_user_all_embeddings, target_item_all_embeddings = self.forward()
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]

        u_embeddings = target_user_all_embeddings[user]
        i_embeddings = target_item_all_embeddings[item]

        scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)
        return scores

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]

        restore_user_e, restore_item_e = self.get_restore_e()
        u_embeddings = restore_user_e[user]
        i_embeddings = restore_item_e[:self.target_num_items]

        scores = torch.matmul(u_embeddings, i_embeddings.transpose(0, 1))
        return scores.view(-1)

    def init_restore_e(self):
        # clear the storage variable when training
        if self.target_restore_user_e is not None or self.target_restore_item_e is not None:
            self.target_restore_user_e, self.target_restore_item_e = None, None

    def get_restore_e(self):
        if self.target_restore_user_e is None or self.target_restore_item_e is None:
            _, _, self.target_restore_user_e, self.target_restore_item_e = self.forward()
        return self.target_restore_user_e, self.target_restore_item_e

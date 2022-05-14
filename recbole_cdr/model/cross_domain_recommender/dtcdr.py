# -*- coding: utf-8 -*-
# @Time   : 2022/3/22
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

r"""
DTCDR
################################################
Reference:
    Feng Zhu et al. "DTCDR: A Framework for Dual-Target Cross-Domain Recommendation." in CIKM 2019.
"""
import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.layers import MLPLayers

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class DTCDR(CrossDomainRecommender):
    r""" This model conduct NeuMF or DMF in both domain where the embedding of overlapped users or items
         are combined from both domain.

        NOTE:This is the simplified version of original DTCDR model.
            To make fair comparison, This implementation only support user ratings in source and target domain.
           Other side information (e.g., user comments, user profiles and item details) is not supported.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DTCDR, self).__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.dropout_prob = config['dropout_prob']
        self.base_model = config['base_model']
        self.alpha = config['alpha']
        assert self.base_model in ['NeuMF', 'DMF'], "based model {} is not supported! ".format(self.base_model)

        # define layers and loss
        if self.base_model == 'NeuMF':
            self.source_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
            self.source_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)

            self.target_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
            self.target_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
            with torch.no_grad():
                self.target_user_embedding.weight[self.target_num_users:].fill_(np.NINF)
                self.target_item_embedding.weight[self.target_num_items:].fill_(np.NINF)

                self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(np.NINF)
                self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(np.NINF)

            self.source_mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
            self.source_mlp_layers.logger = None  # remove logger to use torch.save()
            self.source_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

            self.target_mlp_layers = MLPLayers([2 * self.embedding_size] + self.mlp_hidden_size, self.dropout_prob)
            self.target_mlp_layers.logger = None  # remove logger to use torch.save()
            self.target_predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        else:
            self.source_history_user_id, self.source_history_user_value, _ = dataset.history_user_matrix(domain='source')
            self.source_history_item_id, self.source_history_item_value, _ = dataset.history_item_matrix(domain='source')
            self.source_interaction_matrix = dataset.inter_matrix(form='csr', domain='source').astype(np.float32)
            self.source_history_user_id = self.source_history_user_id.to(self.device)
            self.source_history_user_value = self.source_history_user_value.to(self.device)
            self.source_history_item_id = self.source_history_item_id.to(self.device)
            self.source_history_item_value = self.source_history_item_value.to(self.device)

            self.target_history_user_id, self.target_history_user_value, _ = dataset.history_user_matrix(domain='target')
            self.target_history_item_id, self.target_history_item_value, _ = dataset.history_item_matrix(domain='target')
            self.target_interaction_matrix = dataset.inter_matrix(form='csr', domain='target').astype(np.float32)
            self.target_history_user_id = self.target_history_user_id.to(self.device)
            self.target_history_user_value = self.target_history_user_value.to(self.device)
            self.target_history_item_id = self.target_history_item_id.to(self.device)
            self.target_history_item_value = self.target_history_item_value.to(self.device)

            self.source_user_linear = nn.Linear(in_features=self.source_num_items, out_features=self.embedding_size,
                                                bias=False)
            self.source_item_linear = nn.Linear(in_features=self.source_num_users, out_features=self.embedding_size,
                                                bias=False)
            self.source_user_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)
            self.source_item_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)

            self.target_user_linear = nn.Linear(in_features=self.target_num_items, out_features=self.embedding_size,
                                                bias=False)
            self.target_item_linear = nn.Linear(in_features=self.target_num_users, out_features=self.embedding_size,
                                                bias=False)
            self.target_user_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)
            self.target_item_fc_layers = MLPLayers([self.embedding_size] + self.mlp_hidden_size)

        self.source_sigmoid = nn.Sigmoid()
        self.target_sigmoid = nn.Sigmoid()

        self.loss = nn.BCELoss()
        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)

        return self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))

    def neumf_forward(self, user, item, domain='source'):
        user_source_e = self.source_user_embedding(user)
        user_target_e = self.target_user_embedding(user)
        user_e = torch.maximum(user_source_e, user_target_e)

        item_source_e = self.source_item_embedding(item)
        item_target_e = self.target_item_embedding(item)
        item_e = torch.maximum(item_source_e, item_target_e)

        if domain == 'source':
            output = self.source_sigmoid(self.source_predict_layer(self.source_mlp_layers(torch.cat((user_e, item_e), -1))))
        else:
            output = self.target_sigmoid(self.target_predict_layer(self.target_mlp_layers(torch.cat((user_e, item_e), -1))))
        return output.squeeze(-1)

    def construct_matrix(self, input_tensor, history_id_matrix, history_value_matrix, length):
        col_indices = history_id_matrix[input_tensor].flatten()
        row_indices = torch.arange(input_tensor.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(history_id_matrix.shape[1], dim=0)
        matrix_01 = torch.zeros(1).to(self.device).repeat(input_tensor.shape[0], length)
        matrix_01.index_put_((row_indices, col_indices), history_value_matrix[input_tensor].flatten())
        return matrix_01

    def dmf_forward(self, user, item, domain='source'):
        col_indices = self.source_history_item_id[user].flatten()
        col_indices[col_indices > self.target_num_items] = col_indices[col_indices > self.target_num_items]-(self.target_num_items-self.overlapped_num_items)
        row_indices = torch.arange(user.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(self.source_history_item_id.shape[1], dim=0)
        source_user_matrix = torch.zeros(1).to(self.device).repeat(user.shape[0], self.source_num_items)
        source_user_matrix.index_put_((row_indices, col_indices), self.source_history_item_value[user].flatten())
        source_user_e = self.source_user_linear(source_user_matrix)


        target_user_matrix = self.construct_matrix(user, self.target_history_item_id, self.target_history_item_value, self.target_num_items)
        target_user_e = self.target_user_linear(target_user_matrix)

        user_e = torch.maximum(source_user_e, target_user_e)

        col_indices = self.source_history_user_id[item].flatten()
        col_indices[col_indices > self.target_num_users] = col_indices[col_indices > self.target_num_users] - (
                    self.target_num_users - self.overlapped_num_users)
        row_indices = torch.arange(user.shape[0]).to(self.device)
        row_indices = row_indices.repeat_interleave(self.source_history_user_id.shape[1], dim=0)
        source_item_matrix = torch.zeros(1).to(self.device).repeat(item.shape[0], self.source_num_users)
        source_item_matrix.index_put_((row_indices, col_indices), self.source_history_user_value[user].flatten())
        source_item_e = self.source_item_linear(source_item_matrix)

        target_item_matrix = self.construct_matrix(item, self.target_history_user_id, self.target_history_user_value, self.target_num_users)
        target_item_e = self.target_item_linear(target_item_matrix)

        item_e = torch.maximum(source_item_e, target_item_e)
        
        if domain == 'source':
            user_e = self.source_user_fc_layers(user_e)
            item_e = self.source_item_fc_layers(item_e)
            output = torch.mul(user_e, item_e).sum(dim=1)
            output = self.source_sigmoid(output)
        else:
            user_e = self.target_user_fc_layers(user_e)
            item_e = self.target_item_fc_layers(item_e)
            output = torch.mul(user_e, item_e).sum(dim=1)
            output = self.target_sigmoid(output)
            
        return output

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        if self.base_model == 'NeuMF':
            source_output = self.neumf_forward(source_user, source_item, 'source')
            target_output = self.neumf_forward(target_user, target_item, 'target')

            loss_s = self.loss(source_output, source_label)
            loss_t = self.loss(target_output, target_label)

            return loss_s * self.alpha + loss_t * (1 - self.alpha)
        else:
            source_output = self.dmf_forward(source_user, source_item, 'source')
            target_output = self.dmf_forward(target_user, target_item, 'source')

            loss_s = self.loss(source_output, source_label)
            loss_t = self.loss(target_output, target_label)

            return loss_s * self.alpha + loss_t * (1 - self.alpha)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        if self.base_model == 'NeuMF':
            output = self.neumf_forward(user, item, 'target')
            return output
        else:
            output = self.dmf_forward(user, item, 'target')
            return output

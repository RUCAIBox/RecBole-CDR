# -*- coding: utf-8 -*-
# @Time   : 2022/5/21
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
NATR
################################################
Reference:
    Chen Gao et al. "Cross-domain Recommendation Without Sharing User-relevant Data" in WWW 2019.
"""

import torch
import torch.nn as nn

from recbole.utils import InputType
from recbole.model.loss import RegLoss
from recbole.model.init import xavier_normal_initialization

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class NATR(CrossDomainRecommender):
    r"""NATR propose a neural network method, combining item-level and domain-level
     attention mechanisms to address the challenges in cross-domain learning.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(NATR, self).__init__(config, dataset)

        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "NATR model only support user overlapped or item overlapped dataset! "
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'
        self.phase = None

        # load parameters info
        self.source_embedding_size = config['source_embedding_size']
        self.target_embedding_size = config['target_embedding_size']
        self.reg_weight = config['reg_weight']
        self.max_inter_length = config['max_inter_length']

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        if self.mode == 'overlap_users':
            self.history_user_matrix, self.history_lens, self.mask_mat = self.get_history_user_info(dataset)
        if self.mode == 'overlap_items':
            self.history_item_matrix, self.history_lens, self.mask_mat = self.get_history_item_info(dataset)

        # define layers and loss
        self.source_user_embedding = nn.Embedding(self.total_num_users, self.source_embedding_size)
        self.source_item_embedding = nn.Embedding(self.total_num_items, self.source_embedding_size)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.target_embedding_size)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.target_embedding_size)
        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)
            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
        self.transfer_layer = nn.Linear(self.source_embedding_size, self.target_embedding_size)
        self.unit_attention_layer = nn.Linear(self.target_embedding_size, 1)
        self.domain_attention_layer = nn.Linear(self.target_embedding_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        self.rec_loss = nn.BCELoss()
        self.reg_loss = RegLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def set_phase(self, phase):
        self.phase = phase
        if phase == 'TARGET':
            self.source_item_embedding.weight.requires_grad = False
            self.source_user_embedding.weight.requires_grad = False

    def get_history_item_info(self, dataset):
        history_item_matrix, _, history_lens = dataset.history_item_matrix(domain='target')
        history_item_matrix = history_item_matrix[:, :self.max_inter_length]
        history_item_matrix = history_item_matrix.to(self.device)
        history_lens = history_lens.to(self.device)
        arange_tensor = torch.arange(history_item_matrix.shape[1]).to(self.device)
        mask_mat = (arange_tensor < history_lens.unsqueeze(1)).float()
        return history_item_matrix, history_lens, mask_mat

    def get_history_user_info(self, dataset):
        history_user_matrix, _, history_lens = dataset.history_user_matrix(domain='target')
        history_user_matrix = history_user_matrix[:, :self.max_inter_length]
        history_user_matrix = history_user_matrix.to(self.device)
        history_lens = history_lens.to(self.device)
        arange_tensor = torch.arange(history_user_matrix.shape[1]).to(self.device)
        mask_mat = (arange_tensor < history_lens.unsqueeze(1)).float()
        return history_user_matrix, history_lens, mask_mat

    def phase1_forward(self, user, item):
        user_e = self.source_user_embedding(user)
        item_e = self.source_item_embedding(item)
        score = self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))
        return score

    def calculate_phase1_loss(self, interaction):
        user = interaction[self.SOURCE_USER_ID]
        item = interaction[self.SOURCE_ITEM_ID]
        label = interaction[self.SOURCE_LABEL]
        score = self.phase1_forward(user, item)
        loss = self.rec_loss(score, label)
        return loss

    def phase2_forward(self, user, item):
        user_e = self.target_user_embedding(user)   # (batch, embedding_size)
        item_e = self.target_item_embedding(item)   # (batch, embedding_size)
        if self.mode == 'overlap_items':
            batch_mask_mat = self.mask_mat[user]
            batch_mask_mat = torch.where(batch_mask_mat.bool(), 0., -10000.0)
            history_items = self.history_item_matrix[user]
            history_items_e = self.source_item_embedding(history_items)  # (batch, n_history, source_embedding_size)
            history_items_e = self.transfer_layer(history_items_e)
            unit_attention_score = user_e.unsqueeze(1).expand_as(history_items_e) * history_items_e  # (batch, n_history, embedding_size)
            unit_attention_score = self.unit_attention_layer(self.relu(unit_attention_score)).squeeze(2)
            unit_attention_score += batch_mask_mat
            unit_attention_score = self.softmax(unit_attention_score)   # (batch, n_history)
            unit_attention_score = unit_attention_score.unsqueeze(1)    # (batch, 1, n_history)
            su = torch.bmm(unit_attention_score, history_items_e).squeeze(1)    # (batch, embedding_size)
            pu, qi = user_e, item_e
            b_s = self.domain_attention_layer(self.relu(su * qi))
            b_p = self.domain_attention_layer(self.relu(pu * qi))
            beta_s = torch.exp(b_s) / (torch.exp(b_s) + torch.exp(b_p))
            beta_p = 1 - beta_s
            zu = beta_s * su + beta_p * pu  # (batch, embedding_size)
            score = self.sigmoid(torch.mul(zu, qi).sum(dim=1))
            return score
        elif self.mode == 'overlap_users':
            batch_mask_mat = self.mask_mat[item]
            batch_mask_mat = torch.where(batch_mask_mat.bool(), 0., -10000.0)
            history_users = self.history_user_matrix[item]
            history_users_e = self.source_user_embedding(history_users)
            history_users_e = self.transfer_layer(history_users_e)
            unit_attention_score = item_e.unsqueeze(1).expand_as(history_users_e) * history_users_e  # (batch, n_history, embedding_size)
            unit_attention_score = self.unit_attention_layer(self.relu(unit_attention_score)).squeeze(2)
            unit_attention_score += batch_mask_mat
            unit_attention_score = self.softmax(unit_attention_score)   # (batch, n_history)
            unit_attention_score = unit_attention_score.unsqueeze(1)    # (batch, 1, n_history)
            su = torch.bmm(unit_attention_score, history_users_e).squeeze(1)    # (batch, embedding_size)
            pu, qi = item_e, user_e
            b_s = self.domain_attention_layer(self.relu(su * qi))
            b_p = self.domain_attention_layer(self.relu(pu * qi))
            beta_s = torch.exp(b_s) / (torch.exp(b_s) + torch.exp(b_p))
            beta_p = 1 - beta_s
            zu = beta_s * su + beta_p * pu  # (batch, embedding_size)
            score = self.sigmoid(torch.mul(zu, qi).sum(dim=1))
            return score

    def calculate_phase2_loss(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        label = interaction[self.TARGET_LABEL]
        score = self.phase2_forward(user, item)
        rec_loss = self.rec_loss(score, label)
        reg_loss = self.reg_loss([self.target_user_embedding.weight, self.target_item_embedding.weight,
                                  self.transfer_layer.weight,
                                  self.unit_attention_layer.weight, self.domain_attention_layer.weight])
        loss = rec_loss + self.reg_weight * reg_loss
        return loss

    def calculate_loss(self, interaction):
        if self.phase == 'SOURCE':
            return self.calculate_phase1_loss(interaction)
        elif self.phase == 'TARGET':
            return self.calculate_phase2_loss(interaction)
        else:
            return None

    def predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            score = self.phase1_forward(user, item)
        else:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            score = self.phase2_forward(user, item)
        return score

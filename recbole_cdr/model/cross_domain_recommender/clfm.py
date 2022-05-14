# -*- coding: utf-8 -*-
# @Time   : 2022/3/28
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

r"""
CLFM
################################################
Reference:
    Sheng Gao et al. "Cross-Domain Recommendation via Cluster-Level Latent Factor Model." in PKDD 2013.
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import EmbLoss

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class CLFM(CrossDomainRecommender):
    r"""CLFM factorize the interaction matrix from both domain
        with domain-shared embeddings and domain-specific embeddings.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CLFM, self).__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.user_embedding_size = config['user_embedding_size']
        self.source_item_embedding_size = config['source_item_embedding_size']
        self.target_item_embedding_size = config['source_item_embedding_size']
        self.share_embedding_size = config['share_embedding_size']
        self.alpha = config['alpha']
        self.reg_weight = config['reg_weight']
        assert 0 <= self.share_embedding_size <= self.source_item_embedding_size and \
               0 <= self.share_embedding_size <= self.target_item_embedding_size
        "The number of shared dimension must less than the dimension of both " \
        "the source item embedding and target item embedding"

        # define layers and loss
        self.source_user_embedding = nn.Embedding(self.total_num_users, self.user_embedding_size)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.user_embedding_size)

        self.source_item_embedding = nn.Embedding(self.total_num_items, self.source_item_embedding_size)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.target_item_embedding_size)
        if self.share_embedding_size > 0:
            self.shared_linear = nn.Linear(self.user_embedding_size, self.share_embedding_size, bias=False)
        if self.source_item_embedding_size - self.share_embedding_size > 0:
            self.source_only_linear = \
                nn.Linear(self.user_embedding_size, self.source_item_embedding_size - self.share_embedding_size,
                          bias=False)
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            self.target_only_linear = \
                nn.Linear(self.user_embedding_size, self.target_item_embedding_size - self.share_embedding_size,
                          bias=False)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.source_reg_loss = EmbLoss()
        self.target_reg_loss = EmbLoss()

        self.apply(xavier_normal_initialization)

    def forward(self, user, item):
        pass

    def source_forward(self, user, item):
        user_embedding = self.source_user_embedding(user)
        item_embedding = self.source_item_embedding(item)
        factors = []
        if self.share_embedding_size > 0:
            share_factors = self.shared_linear(user_embedding)
            factors.append(share_factors)
        if self.source_item_embedding_size - self.share_embedding_size > 0:
            only_factors = self.source_only_linear(user_embedding)
            factors.append(only_factors)
        factors = torch.cat(factors, dim=1)
        output = self.sigmoid(torch.mul(factors, item_embedding).sum(dim=1))

        return output

    def target_forward(self, user, item):
        user_embedding = self.target_user_embedding(user)
        item_embedding = self.target_item_embedding(item)
        factors = []
        if self.share_embedding_size > 0:
            share_factors = self.shared_linear(user_embedding)
            factors.append(share_factors)
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            only_factors = self.target_only_linear(user_embedding)
            factors.append(only_factors)
        factors = torch.cat(factors, dim=1)
        output = self.sigmoid(torch.mul(factors, item_embedding).sum(dim=1))
        return output

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        p_source = self.source_forward(source_user, source_item)
        p_target = self.target_forward(target_user, target_item)

        loss_s = self.loss(p_source, source_label) + self.reg_weight * self.source_reg_loss(
            self.source_user_embedding(source_user),
            self.source_item_embedding(source_item))
        loss_t = self.loss(p_target, target_label) + self.reg_weight * self.target_reg_loss(
            self.target_user_embedding(target_user),
            self.target_item_embedding(target_item))

        loss = loss_s * self.alpha + loss_t * (1 - self.alpha)

        return loss

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        p = self.target_forward(user, item)
        return p

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        user_embedding = self.target_user_embedding(user)
        all_item_embedding = self.target_item_embedding.weight[:self.target_num_items]
        factors = []
        if self.share_embedding_size > 0:
            share_factors = self.shared_linear(user_embedding)
            factors.append(share_factors)
        if self.target_item_embedding_size - self.share_embedding_size > 0:
            only_factors = self.target_only_linear(user_embedding)
            factors.append(only_factors)
        factors = torch.cat(factors, dim=1)
        score = torch.matmul(factors, all_item_embedding.transpose(0, 1))
        return score.view(-1)

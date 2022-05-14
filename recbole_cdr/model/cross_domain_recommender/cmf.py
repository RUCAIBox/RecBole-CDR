# -*- coding: utf-8 -*-
# @Time   : 2022/3/8
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

r"""
CMF
################################################
Reference:
    Ajit P. Singh et al. "Relational Learning via Collective Matrix Factorization." in SIGKDD 2008.
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
from recbole.model.loss import EmbLoss

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class CMF(CrossDomainRecommender):
    r""" CMF jointly factorize the interaction matrix from both domain
        with mapping the same user (or item) to one vector.
        In this implementation, we set alpha to control the loss from two domains.
    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CMF, self).__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.alpha = config['alpha']
        self.lamda = config['lambda']
        self.gamma = config['gamma']

        # define layers and loss
        self.user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.source_reg_loss = EmbLoss()
        self.target_reg_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def get_user_embedding(self, user):
        r""" Get a batch of user embedding tensor according to input user's id.

        Args:
            user (torch.LongTensor): The input tensor that contains user's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of user, shape: [batch_size, embedding_size]
        """
        return self.user_embedding(user)

    def get_item_embedding(self, item):
        r""" Get a batch of item embedding tensor according to input item's id.

        Args:
            item (torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            torch.FloatTensor: The embedding tensor of a batch of item, shape: [batch_size, embedding_size]
        """
        return self.item_embedding(item)

    def forward(self, user, item):
        user_e = self.get_user_embedding(user)
        item_e = self.get_item_embedding(item)

        return self.sigmoid(torch.mul(user_e, item_e).sum(dim=1))

    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        p_source = self.forward(source_user, source_item)
        p_target = self.forward(target_user, target_item)

        loss_s = self.loss(p_source, source_label) + \
                 self.lamda * self.source_reg_loss(self.get_user_embedding(source_user),
                                                   self.get_item_embedding(source_item))
        loss_t = self.loss(p_target, target_label) + \
                 self.gamma * self.source_reg_loss(self.get_user_embedding(target_user),
                                                   self.get_item_embedding(target_item))
        return loss_s * self.alpha + loss_t * (1 - self.alpha)

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        p = self.forward(user, item)
        return p

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        user_e = self.get_user_embedding(user)
        all_item_e = self.item_embedding.weight[:self.target_num_items]
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

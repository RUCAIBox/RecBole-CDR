# -*- coding: utf-8 -*-
# @Time   : 2022/3/29
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

r"""
DeepAPF
################################################
Reference:
    Huan Yan et al. "DeepAPF: Deep Attentive Probabilistic Factorization for Multi-site Video Recommendation."
    in IJCAI 2019.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class DeepAPF(CrossDomainRecommender):
    r"""It decomposes the embedding into common part and specific part with attention mechanism to merge.
    We extend the basic DeepAPF model in a symmetrical way to support those datasets that have overlapped items.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(DeepAPF, self).__init__(config, dataset)
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "DeepAPF model only support user overlapped or item overlapped dataset! "
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'

        self.embedding_size = config['embedding_size']
        self.beta = config['beta']

        self.source_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.share_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)

        self.source_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        self.share_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)

        self.user_mlp = self.seq = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size), nn.ReLU(),
            nn.Linear(self.embedding_size, 1, bias=False))

        self.item_mlp = self.seq = nn.Sequential(
            nn.Linear(self.embedding_size, self.embedding_size), nn.ReLU(),
            nn.Linear(self.embedding_size, 1, bias=False))

        self.predict_layer = nn.Linear(self.embedding_size, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

        self.apply(xavier_normal_initialization)

    def source_forward(self, user, item):
        if self.mode == 'overlap_users':
            share_user_embedding = self.share_user_embedding(user)
            source_only_user_embedding = self.source_user_embedding(user)
            item_embedding = self.source_item_embedding(item)
            mask_tensor = (user > self.overlapped_num_users).unsqueeze(-1)

            alpha_share = self.user_mlp(torch.mul(share_user_embedding, item_embedding))
            alpha_source_only = self.user_mlp(torch.mul(source_only_user_embedding, item_embedding))

            alpha_share = alpha_share.masked_fill(mask_tensor, value=torch.tensor(-1e31))

            alpha = torch.cat([alpha_share, alpha_source_only], dim=1)
            alpha = F.softmax(alpha, dim=1).unsqueeze(1)

            user_embedding = alpha * torch.cat([share_user_embedding.unsqueeze(2),
                                                source_only_user_embedding.unsqueeze(2)], dim=2)
            user_embedding = user_embedding.sum(dim=2)
            output = self.sigmoid(self.predict_layer(torch.mul(user_embedding, item_embedding)))

        else:
            user_embedding = self.source_user_embedding(user)

            share_item_embedding = self.share_item_embedding(item)
            source_only_item_embedding = self.source_item_embedding(item)
            mask_tensor = (item > self.overlapped_num_items).unsqueeze(-1)

            alpha_share = self.item_mlp(torch.mul(share_item_embedding, user_embedding))
            alpha_source_only = self.item_mlp(torch.mul(source_only_item_embedding, user_embedding))

            alpha_share = alpha_share.masked_fill(mask_tensor, value=torch.tensor(-1e31))

            alpha = torch.cat([alpha_share, alpha_source_only], dim=1)
            alpha = F.softmax(alpha, dim=1).unsqueeze(1)

            item_embedding = alpha * torch.cat([share_item_embedding.unsqueeze(2),
                                                source_only_item_embedding.unsqueeze(2)], dim=2)
            item_embedding = item_embedding.sum(dim=2)
            output = self.sigmoid(self.predict_layer(torch.mul(user_embedding, item_embedding)))
        return output.squeeze(-1)
    
    def target_forward(self, user, item):
        if self.mode == 'overlap_users':
            share_user_embedding = self.share_user_embedding(user)
            target_only_user_embedding = self.target_user_embedding(user)
            item_embedding = self.target_item_embedding(item)
            mask_tensor = (user > self.overlapped_num_users).unsqueeze(-1)

            alpha_share = self.user_mlp(torch.mul(share_user_embedding, item_embedding))
            alpha_target_only = self.user_mlp(torch.mul(target_only_user_embedding, item_embedding))

            alpha_share = alpha_share.masked_fill(mask_tensor, value=torch.tensor(-1e31))

            alpha = torch.cat([alpha_share, alpha_target_only], dim=1)
            alpha = F.softmax(alpha, dim=1).unsqueeze(1)

            user_embedding = alpha * torch.cat([share_user_embedding.unsqueeze(2),
                                                target_only_user_embedding.unsqueeze(2)], dim=2)
            user_embedding = user_embedding.sum(dim=2)
            output = self.sigmoid(self.predict_layer(torch.mul(user_embedding, item_embedding)))

        else:
            user_embedding = self.target_user_embedding(user)

            share_item_embedding = self.share_item_embedding(item)
            target_only_item_embedding = self.target_item_embedding(item)
            mask_tensor = (item > self.overlapped_num_items).unsqueeze(-1)

            alpha_share = self.item_mlp(torch.mul(share_item_embedding, user_embedding))
            alpha_target_only = self.item_mlp(torch.mul(target_only_item_embedding, user_embedding))

            alpha_share = alpha_share.masked_fill(mask_tensor, value=torch.tensor(-1e31))

            alpha = torch.cat([alpha_share, alpha_target_only], dim=1)
            alpha = F.softmax(alpha, dim=1).unsqueeze(1)

            item_embedding = alpha * torch.cat([share_item_embedding.unsqueeze(2),
                                                target_only_item_embedding.unsqueeze(2)], dim=2)
            item_embedding = item_embedding.sum(dim=2)
            output = self.sigmoid(self.predict_layer(torch.mul(user_embedding, item_embedding)))
        return output.squeeze(-1)

    def forward(self):
        pass

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        p = self.target_forward(user, item)
        return p
        
    def calculate_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_item = interaction[self.SOURCE_ITEM_ID]
        source_label = interaction[self.SOURCE_LABEL]

        target_user = interaction[self.TARGET_USER_ID]
        target_item = interaction[self.TARGET_ITEM_ID]
        target_label = interaction[self.TARGET_LABEL]

        p_source = self.source_forward(source_user, source_item)
        p_target = self.target_forward(target_user, target_item)

        loss_s = self.loss(p_source, source_label)
        loss_t = self.loss(p_target, target_label)

        return loss_s + loss_t

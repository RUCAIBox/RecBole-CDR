# -*- coding: utf-8 -*-
# @Time   : 2022/4/8
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com

r"""
EMCDR
################################################
Reference:
    Tong Man et al. "Cross-Domain Recommendation: An Embedding and Mapping Approach" in IJCAI 2017.
"""

import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.model.loss import BPRLoss
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class EMCDR(CrossDomainRecommender):
    r"""EMCDR learns an mapping function from source latent space
        to target latent space.

    """

    def __init__(self, config, dataset):
        super(EMCDR, self).__init__(config, dataset)

        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "EMCDR model only support user overlapped or item overlapped dataset! "
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'
        self.phase = 'both'

        # load parameters info
        self.latent_factor_model = config['latent_factor_model']
        if self.latent_factor_model == 'MF':
            input_type = InputType.POINTWISE
            # load dataset info
            self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
            self.TARGET_LABEL = dataset.target_domain_dataset.label_field
            self.loss = nn.MSELoss()
        else:
            input_type = InputType.PAIRWISE
            # load dataset info
            self.loss = BPRLoss()
        self.source_latent_dim = config['source_embedding_size']  # int type:the embedding size of source latent space
        self.target_latent_dim = config['target_embedding_size']  # int type:the embedding size of target latent space
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.map_func = config['mapping_function']
        if self.map_func == 'linear':
            self.mapping = nn.Linear(self.source_latent_dim, self.target_latent_dim, bias=False)
        else:
            assert config["mlp_hidden_size"] is not None
            mlp_layers_dim = [self.source_latent_dim] + config["mlp_hidden_size"] + [self.target_latent_dim]
            self.mapping = self.mlp_layers(mlp_layers_dim)

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(self.total_num_users, self.source_latent_dim)
        self.source_item_embedding = torch.nn.Embedding(self.total_num_items, self.source_latent_dim)

        self.target_user_embedding = torch.nn.Embedding(self.total_num_users, self.target_latent_dim)
        self.target_item_embedding = torch.nn.Embedding(self.total_num_items, self.target_latent_dim)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.reg_loss = EmbLoss()
        self.map_loss = nn.MSELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def mlp_layers(self, layer_dims):
        mlp_modules = []
        for i, (d_in, d_out) in enumerate(zip(layer_dims[:-1], layer_dims[1:])):
            mlp_modules.append(nn.Linear(d_in, d_out))
            if i != len(layer_dims[:-1]) - 1:
                mlp_modules.append(nn.Tanh())

        return nn.Sequential(*mlp_modules)

    def set_phase(self, phase):
        self.phase = phase

    def source_forward(self, user, item):
        user_e = self.source_user_embedding(user)
        item_e = self.source_item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)

    def target_forward(self, user, item):
        user_e = self.target_user_embedding(user)
        item_e = self.target_item_embedding(item)

        return torch.mul(user_e, item_e).sum(dim=1)

    def calculate_source_loss(self, interaction):
        if self.latent_factor_model == 'MF':
            source_user = interaction[self.SOURCE_USER_ID]
            source_item = interaction[self.SOURCE_ITEM_ID]
            source_label = interaction[self.SOURCE_LABEL]

            p_source = self.source_forward(source_user, source_item)

            loss_s = self.loss(p_source, source_label) + \
                     self.reg_weight * self.reg_loss(self.source_user_embedding(source_user),
                                                     self.source_item_embedding(source_item))
        else:
            source_user = interaction[self.SOURCE_USER_ID]
            source_pos_item = interaction[self.SOURCE_ITEM_ID]
            source_neg_item = interaction[self.SOURCE_NEG_ITEM_ID]

            pos_item_score = self.source_forward(source_user, source_pos_item)
            neg_item_score = self.source_forward(source_user, source_neg_item)
            loss_s = self.loss(pos_item_score, neg_item_score) + \
                     self.reg_weight * self.reg_loss(self.source_user_embedding(source_user),
                                                     self.source_item_embedding(source_pos_item))
        return loss_s

    def calculate_target_loss(self, interaction):
        if self.latent_factor_model == 'MF':
            target_user = interaction[self.TARGET_USER_ID]
            target_item = interaction[self.TARGET_ITEM_ID]
            target_label = interaction[self.TARGET_LABEL]

            p_target = self.target_forward(target_user, target_item)

            loss_t = self.loss(p_target, target_label) + \
                     self.reg_weight * self.reg_loss(self.target_user_embedding(target_user),
                                                     self.target_item_embedding(target_item))
        else:
            target_user = interaction[self.TARGET_USER_ID]
            target_pos_item = interaction[self.TARGET_ITEM_ID]
            target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]

            pos_item_score = self.target_forward(target_user, target_pos_item)
            neg_item_score = self.target_forward(target_user, target_neg_item)
            loss_t = self.loss(pos_item_score, neg_item_score) + \
                     self.reg_weight * self.reg_loss(self.target_user_embedding(target_user),
                                                     self.target_item_embedding(target_pos_item))
        return loss_t

    def calculate_map_loss(self, interaction):
        idx = interaction[self.OVERLAP_ID]
        if self.mode == 'overlap_users':
            source_user_e = self.source_user_embedding(idx)
            target_user_e = self.target_user_embedding(idx)
            map_e = self.mapping(source_user_e)
            loss = self.map_loss(map_e, target_user_e)
        else:
            source_item_e = self.source_item_embedding(idx)
            target_item_e = self.target_item_embedding(idx)
            map_e = self.mapping(source_item_e)
            loss = self.map_loss(map_e, target_item_e)
        return loss

    def calculate_loss(self, interaction):
        if self.phase == 'SOURCE':
            return self.calculate_source_loss(interaction)
        elif self.phase == 'OVERLAP':
            return self.calculate_map_loss(interaction)
        else:
            return self.calculate_target_loss(interaction)

    def predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            user_e = self.source_user_embedding(user)
            item_e = self.source_item_embedding(item)
            score = torch.mul(user_e, item_e).sum(dim=1)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            user_e = self.target_user_embedding(user)
            item_e = self.target_item_embedding(item)
            score = torch.mul(user_e, item_e).sum(dim=1)
        else:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            if self.mode == 'overlap_users':
                repeat_user = user.repeat(self.source_latent_dim, 1).transpose(0, 1)
                user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping(self.source_user_embedding(user)),
                                     self.target_user_embedding(user))
                item_e = self.target_item_embedding(item)
            else:
                user_e = self.target_user_embedding(user)
                repeat_item = item.repeat(self.source_latent_dim, 1).transpose(0, 1)
                item_e = torch.where(repeat_item < self.overlapped_num_items, self.mapping(self.source_item_embedding(item)),
                                     self.target_item_embedding(item))

            score = torch.mul(user_e, item_e).sum(dim=1)
        return score

    def full_sort_predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            user_e = self.source_user_embedding(user)
            overlap_item_e = self.source_item_embedding.weight[:self.overlapped_num_items]
            source_item_e = self.source_item_embedding.weight[self.target_num_items:]
            all_item_e = torch.cat([overlap_item_e, source_item_e], dim=0)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            user_e = self.target_user_embedding(user)
            all_item_e = self.target_item_embedding.weight[:self.target_num_items]
        else:
            user = interaction[self.TARGET_USER_ID]
            if self.mode == 'overlap_users':
                repeat_user = user.repeat(self.source_latent_dim, 1).transpose(0, 1)
                user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping(self.source_user_embedding(user)),
                                     self.target_user_embedding(user))
                all_item_e = self.target_item_embedding.weight[:self.target_num_items]
            else:
                user_e = self.target_user_embedding(user)
                overlap_item_e = self.mapping(self.source_item_embedding.weight[:self.overlapped_num_items])
                target_item_e = self.target_item_embedding.weight[self.overlapped_num_items:self.target_num_items]
                all_item_e = torch.cat([overlap_item_e, target_item_e], dim=0)

        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score.view(-1)

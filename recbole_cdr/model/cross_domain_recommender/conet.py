# -*- coding: utf-8 -*-
# @Time   : 2022/3/30
# @Author : Gaowei Zhang
# @Email  : 1462034631@qq.com
# UPDATE
# @Time    :   2022/4/11
# @Author  :   Zihan Lin
# @email   :   zhlin@ruc.edu.cn

r"""
CoNet
################################################
Reference:
    Guangneng Hu et al. "CoNet: Collaborative Cross Networks for Cross-Domain Recommendation." in CIKM 2018.
"""

import torch
import torch.nn as nn

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType


class CoNet(CrossDomainRecommender):
    r"""CoNet takes neural network as the basic model and uses cross connections
        unit to improve the learning of matching functions in the current domain.

    """
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(CoNet, self).__init__(config, dataset)

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field

        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "CoNet model only support user overlapped or item overlapped dataset! "
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'

        # load parameters info
        self.device = config['device']

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalization
        self.cross_layers = config["mlp_hidden_size"]  # list type: the list of hidden layers size

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)
        self.target_user_embedding = torch.nn.Embedding(num_embeddings=self.total_num_users, embedding_dim=self.latent_dim)

        self.source_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)
        self.target_item_embedding = torch.nn.Embedding(num_embeddings=self.total_num_items, embedding_dim=self.latent_dim)

        self.loss = nn.BCELoss()

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.source_crossunit_linear, self.source_crossunit_act \
            = self.cross_units([2 * self.latent_dim] + self.cross_layers)
        self.source_outputunit = nn.Sequential(
            nn.Linear(self.cross_layers[-1], 1),
            nn.Sigmoid()
        )

        self.target_crossunit_linear, self.target_crossunit_act \
            = self.cross_units([2 * self.latent_dim] + self.cross_layers)
        self.target_outputunit = nn.Sequential(
            nn.Linear(self.cross_layers[-1], 1),
            nn.Sigmoid()
        )

        self.crossparas = self.cross_parameters([2 * self.latent_dim] + self.cross_layers)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def cross_units(self, cross_layers):
        cross_modules_linear, cross_modules_act = [], []
        for i, (d_in, d_out) in enumerate(zip(cross_layers[:-1], cross_layers[1:])):
            cross_modules_linear.append(nn.Linear(d_in, d_out))
            cross_modules_act.append(nn.ReLU())
        return nn.ModuleList(cross_modules_linear), nn.ModuleList(cross_modules_act)

    def cross_parameters(self, cross_layers):
        cross_paras = []
        for i, (d_in, d_out) in enumerate(zip(cross_layers[:-1], cross_layers[1:])):
            para = nn.Linear(d_in, d_out, bias=False)
            cross_paras.append(para)
        return nn.ModuleList(cross_paras)

    def source_forward(self, user, item):
        source_user_embedding = self.source_user_embedding(user)
        source_item_embedding = self.source_item_embedding(item)
        target_user_embedding = self.target_user_embedding(user)
        target_item_embedding = self.target_item_embedding(item)
        source_crossinput = torch.cat([source_user_embedding, source_item_embedding], dim=1).to(self.device)
        target_crossinput = torch.cat([target_user_embedding, target_item_embedding], dim=1).to(self.device)

        if self.mode == 'overlap_users':
            overlap_idx = user < self.overlapped_num_users
        else:
            overlap_idx = item < self.overlapped_num_items

        for i in range(len(self.source_crossunit_linear)):
            source_fc_module, source_act_module = self.source_crossunit_linear[i], self.source_crossunit_act[i]
            source_fc_module = source_fc_module
            source_act_module = source_act_module
            cross_para = self.crossparas[i].weight.t()
            target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]
            target_fc_module = target_fc_module
            target_act_module = target_act_module

            source_crossoutput = source_fc_module(source_crossinput)
            source_crossoutput[overlap_idx] = source_crossoutput[overlap_idx] + torch.mm(target_crossinput, cross_para)[
                overlap_idx]
            source_crossoutput = source_act_module(source_crossoutput)

            target_crossoutput = target_fc_module(target_crossinput)
            target_crossoutput[overlap_idx] = target_crossoutput[overlap_idx] + torch.mm(source_crossinput, cross_para)[
                overlap_idx]
            target_crossoutput = target_act_module(target_crossoutput)

            source_crossinput = source_crossoutput
            target_crossinput = target_crossoutput

        source_out = self.source_outputunit(source_crossinput).squeeze()

        return source_out

    def target_forward(self, user, item):
        source_user_embedding = self.source_user_embedding(user)
        source_item_embedding = self.source_item_embedding(item)
        target_user_embedding = self.target_user_embedding(user)
        target_item_embedding = self.target_item_embedding(item)
        source_crossinput = torch.cat([source_user_embedding, source_item_embedding], dim=1).to(self.device)
        target_crossinput = torch.cat([target_user_embedding, target_item_embedding], dim=1).to(self.device)

        if self.mode == 'overlap_users':
            overlap_idx = user < self.overlapped_num_users
        else:
            overlap_idx = item < self.overlapped_num_items

        for i in range(len(self.target_crossunit_linear)):
            source_fc_module, source_act_module = self.source_crossunit_linear[i], self.source_crossunit_act[i]
            source_fc_module = source_fc_module
            source_act_module = source_act_module
            cross_para = self.crossparas[i].weight.t()
            target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]
            target_fc_module = target_fc_module
            target_act_module = target_act_module

            source_crossoutput = source_fc_module(source_crossinput)
            source_crossoutput[overlap_idx] = source_crossoutput[overlap_idx] + torch.mm(target_crossinput, cross_para)[
                overlap_idx]
            source_crossoutput = source_act_module(source_crossoutput)

            target_crossoutput = target_fc_module(target_crossinput)
            target_crossoutput[overlap_idx] = target_crossoutput[overlap_idx] + torch.mm(source_crossinput, cross_para)[
                overlap_idx]
            target_crossoutput = target_act_module(target_crossoutput)

            source_crossinput = source_crossoutput
            target_crossinput = target_crossoutput

        target_out = self.target_outputunit(target_crossinput).squeeze()

        return target_out

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

        reg_loss = 0
        for para in self.crossparas:
            reg_loss += torch.norm(para.weight)
        loss = loss_s + loss_t + reg_loss

        return loss

    def predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        item = interaction[self.TARGET_ITEM_ID]
        user_e = self.target_user_embedding(user)
        item_e = self.target_item_embedding(item)
        input = torch.cat([user_e, item_e], dim=1)

        for i in range(len(self.target_crossunit_linear)):
            target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]
            output = target_act_module(target_fc_module(input))

            input = output

        p = self.target_outputunit(input)

        return p

    def full_sort_predict(self, interaction):
        user = interaction[self.TARGET_USER_ID]
        user_e = self.target_user_embedding(user)
        user_num = user_e.shape[0]
        all_item_e = self.target_item_embedding.weight[:self.target_num_items]
        item_num = all_item_e.shape[0]
        all_user_e = user_e.repeat(1, item_num).view(-1, self.latent_dim)
        user_e_list = torch.split(all_user_e, [item_num]*user_num)
        score_list = []
        for u_embed in user_e_list:
            input = torch.cat([u_embed, all_item_e], dim=1)
            for i in range(len(self.target_crossunit_linear)):
                target_fc_module, target_act_module = self.target_crossunit_linear[i], self.target_crossunit_act[i]
                output = target_act_module(target_fc_module(input))

                input = output

            p = self.target_outputunit(input)
            score_list.append(p)
        score = torch.cat(score_list, dim=1).transpose(0, 1)
        return score

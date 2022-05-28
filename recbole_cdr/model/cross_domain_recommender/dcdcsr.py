# -*- coding: utf-8 -*-
# @Time   : 2022/5/21
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
DCDCSR
################################################
Reference:
    Feng Zhu et al. "A Deep Framework for Cross-Domain and Cross-System Recommendations" in IJCAI 2018.
"""

import torch
import torch.nn as nn
import numpy as np

from recbole.utils import InputType
from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.model.loss import BPRLoss

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class DCDCSR(CrossDomainRecommender):
    r"""DCDCSR utilizes the sparsity degrees of individual users and items
        in the source and target domains to learn an mapping function from source
        latent space to target latent space.

    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(DCDCSR, self).__init__(config, dataset)

        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "DCDCSR model only support user overlapped or item overlapped dataset! "
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'
        self.phase = None
        self.phase2count = {'SOURCE': 0, 'TARGET': 0, 'BOTH': 0, 'OVERLAP': 0}

        # load parameters info
        self.latent_factor_model = config['latent_factor_model']
        assert self.latent_factor_model in ['BPR'],\
            "latent_factor model must be in [BPR]"
        self.embedding_size = config['embedding_size']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.k = config['k']
        self.map_batch_size = config['map_batch_size']

        # load dataset info
        self.SOURCE_LABEL = dataset.source_domain_dataset.label_field
        self.TARGET_LABEL = dataset.target_domain_dataset.label_field
        if self.mode == 'overlap_items':
            self.source_item2pop = self.build_unit2pop(dataset, unit='item', domain='source')
            self.target_item2pop = self.build_unit2pop(dataset, unit='item', domain='target')
        elif self.mode == 'overlap_users':
            self.source_user2pop = self.build_unit2pop(dataset, unit='user', domain='source')
            self.target_user2pop = self.build_unit2pop(dataset, unit='user', domain='target')

        # define layers and loss
        self.source_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.source_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        self.target_user_embedding = nn.Embedding(self.total_num_users, self.embedding_size)
        self.target_item_embedding = nn.Embedding(self.total_num_items, self.embedding_size)
        self.benchmark_embedding = None
        self.affine_embedding = None
        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)
        self.mapping_mlp_layers = MLPLayers(layers=[self.embedding_size] + self.mlp_hidden_size + [self.embedding_size],
                                            activation='tanh', dropout=0, bn=False)

        self.rec_loss = None
        if self.latent_factor_model == 'BPR':
            self.rec_loss = BPRLoss()
        self.map_loss = nn.MSELoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)

    @staticmethod
    def build_unit2pop(dataset, unit='user', domain='source'):
        if unit == 'user':
            _, _, history_lens = dataset.history_item_matrix(domain=domain)
        else:
            _, _, history_lens = dataset.history_user_matrix(domain=domain)
        return history_lens.float()

    def set_phase(self, phase):
        self.phase = phase
        self.phase2count[phase] += 1
        if self.phase == 'BOTH':
            self.build_benchmark_embedding()
        if self.phase == 'TARGET' and self.phase2count[self.phase] == 2:
            if self.mode == 'overlap_users':
                target_user_embedding, mean_, max_ = \
                    self.maxmin_normalize(self.target_user_embedding.weight[:self.target_num_users])
                self.affine_embedding = self.mapping_mlp_layers(
                    target_user_embedding)
                self.affine_embedding = self.affine_embedding * (max_ - mean_) + mean_
                self.affine_embedding = self.affine_embedding.detach()
            elif self.mode == 'overlap_items':
                target_item_embedding, mean_, max_ = \
                    self.maxmin_normalize(self.target_item_embedding.weight[:self.target_num_items])
                self.affine_embedding = self.mapping_mlp_layers(
                    target_item_embedding)
                self.affine_embedding = self.affine_embedding * (max_ - mean_) + mean_
                self.affine_embedding = self.affine_embedding.detach()

    def calculate_rec_loss(self, interaction, user_embeds, item_embeds,
                           user_field, item_field, neg_item_field, label_field):
        loss = None
        if self.latent_factor_model == 'BPR':
            user = interaction[user_field]
            item = interaction[item_field]
            neg_item = interaction[neg_item_field]

            user_e = user_embeds[user]
            item_e = item_embeds[item]
            neg_item_e = item_embeds[neg_item]
            pos_score = torch.mul(user_e, item_e).sum(dim=1)
            neg_score = torch.mul(user_e, neg_item_e).sum(dim=1)
            loss = self.rec_loss(pos_score, neg_score)

        return loss

    def build_unit_benchmark_embedding(self, total_num_units, overlapped_num_units,
                                       source_unit2pop, target_unit2pop,
                                       source_unit_embeddings, target_unit_embeddings):
        self.benchmark_embedding = torch.FloatTensor(total_num_units, self.embedding_size).to(self.device)
        for idx in range(overlapped_num_units):
            denominator = source_unit2pop[idx] + target_unit2pop[idx]
            if denominator == 0:
                denominator = 1
            alpha_s = source_unit2pop[idx] / denominator
            alpha_t = 1 - alpha_s
            self.benchmark_embedding[idx] = (alpha_s * target_unit_embeddings[idx] +
                                             alpha_t * source_unit_embeddings[idx]).detach()
        for idx in range(overlapped_num_units, total_num_units):
            i_e = target_unit_embeddings[idx]     # (embedding_size)
            sim_i = torch.mm(source_unit_embeddings, i_e.unsqueeze(1)).squeeze(1)   # (n_items)
            sim, index = torch.topk(sim_i, k=self.k, dim=0)     # (k)
            sn = torch.mean(source_unit2pop[index])
            beta_t_i = sn / (sn + target_unit2pop[idx])
            sim_e = source_unit_embeddings[index]   # (k, embedding_size)
            sim_e = torch.mm(sim.unsqueeze(0), sim_e).squeeze(0)    # (embedding_size)
            sum_sim = torch.sum(sim) if torch.sum(sim) > 0 else 1
            sim_e = sim_e / sum_sim
            self.benchmark_embedding[idx] = ((1 - beta_t_i) * target_unit_embeddings[idx] +
                                            beta_t_i * sim_e).detach()

    def build_benchmark_embedding(self):
        if self.mode == 'overlap_users':
            self.build_unit_benchmark_embedding(self.total_num_users, self.overlapped_num_users,
                                                self.source_user2pop, self.target_user2pop,
                                                self.source_user_embedding.weight[:self.overlapped_num_users], self.target_user_embedding.weight)
        elif self.mode == 'overlap_items':
            self.build_unit_benchmark_embedding(self.total_num_items, self.overlapped_num_items,
                                                self.source_item2pop, self.target_item2pop,
                                                self.source_item_embedding.weight[:self.overlapped_num_items], self.target_item_embedding.weight)


    def maxmin_normalize(self, embed_weight):
        min_ = torch.amin(embed_weight, dim=1, keepdim=True)
        max_ = torch.amax(embed_weight, dim=1, keepdim=True)
        mean_ = (max_ + min_) / 2
        normal_mat = (embed_weight - mean_) / (max_ - mean_)
        return normal_mat, mean_, max_

    def calculate_unit_map_loss(self, target_num_units, target_unit_embeddings):
        sampled_index = np.random.randint(0, target_num_units, self.map_batch_size)
        item_embeddings = target_unit_embeddings[sampled_index]
        item_embeddings, _, _ = self.maxmin_normalize(item_embeddings)
        item_embeddings = self.mapping_mlp_layers(item_embeddings)
        benchmark_embeddings = self.benchmark_embedding[sampled_index]
        benchmark_embeddings, _, _ = self.maxmin_normalize(benchmark_embeddings)
        loss = self.map_loss(item_embeddings, benchmark_embeddings)
        return loss

    def calculate_map_loss(self):
        loss = None
        if self.mode == 'overlap_users':
            loss = self.calculate_unit_map_loss(self.target_num_users, self.target_user_embedding.weight)
        elif self.mode == 'overlap_items':
            loss = self.calculate_unit_map_loss(self.target_num_items, self.target_item_embedding.weight)
        return loss

    def calculate_loss(self, interaction):
        if self.phase == 'SOURCE' and self.phase2count[self.phase] == 1:
            return self.calculate_rec_loss(
                interaction, self.source_user_embedding.weight, self.source_item_embedding.weight,
                self.SOURCE_USER_ID, self.SOURCE_ITEM_ID, self.SOURCE_NEG_ITEM_ID, self.SOURCE_LABEL)
        elif self.phase == 'TARGET' and self.phase2count[self.phase] == 1:
            return self.calculate_rec_loss(
                interaction, self.target_user_embedding.weight, self.target_item_embedding.weight,
                self.TARGET_USER_ID, self.TARGET_ITEM_ID, self.TARGET_NEG_ITEM_ID, self.TARGET_LABEL)
        elif self.phase == 'BOTH':
            return self.calculate_map_loss()
        elif self.phase == 'TARGET' and self.phase2count[self.phase] == 2:
            if self.mode == 'overlap_users':
                return self.calculate_rec_loss(
                    interaction, self.affine_embedding, self.target_item_embedding.weight,
                    self.TARGET_USER_ID, self.TARGET_ITEM_ID, self.TARGET_NEG_ITEM_ID, self.TARGET_LABEL)
            elif self.mode == 'overlap_items':
                return self.calculate_rec_loss(
                    interaction, self.target_user_embedding.weight, self.affine_embedding,
                    self.TARGET_USER_ID, self.TARGET_ITEM_ID, self.TARGET_NEG_ITEM_ID, self.TARGET_LABEL)

    def full_sort_predict(self, interaction):
        user_e, all_item_e = None, None
        if self.phase == 'SOURCE' and self.phase2count[self.phase] == 1:
            user = interaction[self.SOURCE_USER_ID]
            user_e = self.source_user_embedding(user)
            all_item_e1 = self.source_item_embedding.weight[:self.overlapped_num_items]
            all_item_e2 = self.source_item_embedding.weight[self.target_num_items:]
            all_item_e = torch.cat([all_item_e1, all_item_e2], dim=0)
        elif self.phase == 'TARGET' and self.phase2count[self.phase] == 1:
            user = interaction[self.TARGET_USER_ID]
            user_e = self.target_user_embedding(user)
            all_item_e = self.target_item_embedding.weight[:self.target_num_items]
        elif self.phase == 'TARGET' and self.phase2count[self.phase] == 2:
            user = interaction[self.TARGET_USER_ID]
            if self.mode == 'overlap_users':
                user_e = self.affine_embedding[user]
                all_item_e = self.target_item_embedding.weight[:self.target_num_items]
            elif self.mode == 'overlap_items':
                user_e = self.target_user_embedding(user)
                all_item_e = self.affine_embedding
        else:
            user = interaction[self.TARGET_USER_ID]
            if self.mode == 'overlap_users':
                user_e = self.affine_embedding[user]
                all_item_e = self.target_item_embedding.weight[:self.target_num_items]
            elif self.mode == 'overlap_items':
                user_e = self.target_user_embedding(user)
                all_item_e = self.affine_embedding
        score = torch.matmul(user_e, all_item_e.transpose(0, 1))
        return score

    def predict(self, interaction):
        user_e, item_e = None, None
        if self.phase == 'SOURCE' and self.phase2count[self.phase] == 1:
            user = interaction[self.SOURCE_USER_ID]
            item = interaction[self.SOURCE_ITEM_ID]
            user_e = self.source_user_embedding(user)
            item_e = self.source_item_embedding(item)
        elif self.phase == 'TARGET' and self.phase2count[self.phase] == 1:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            user_e = self.target_user_embedding(user)
            item_e = self.target_item_embedding(item)
        elif self.phase == 'TARGET' and self.phase2count[self.phase] == 2:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            if self.mode == 'overlap_users':
                user_e = self.affine_embedding[user]
                item_e = self.target_item_embedding(item)
            elif self.mode == 'overlap_items':
                user_e = self.target_user_embedding(user)
                item_e = self.affine_embedding[item]
        else:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            if self.mode == 'overlap_users':
                user_e = self.affine_embedding[user]
                item_e = self.target_item_embedding(item)
            elif self.mode == 'overlap_items':
                user_e = self.target_user_embedding(user)
                item_e = self.affine_embedding[item]
        score = torch.mul(user_e, item_e).sum(dim=1)
        return score

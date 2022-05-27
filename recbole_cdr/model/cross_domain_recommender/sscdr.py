# -*- coding: utf-8 -*-
# @Time   : 2022/5/13
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

r"""
SSCDR
################################################
Reference:
    SeongKu Kang et al. "Semi-Supervised Learning for Cross-Domain Recommendation to Cold-Start Users" in CIKM 2019.
"""
import numpy as np
import torch
import torch.nn as nn

from recbole.model.init import xavier_normal_initialization
from recbole.model.layers import MLPLayers
from recbole.utils import InputType

from recbole_cdr.model.crossdomain_recommender import CrossDomainRecommender


class SSCDR(CrossDomainRecommender):
    r"""SSCDR conducts the embedding mapping by both supervised way and semi-supervised way.
        In this implementation, the mapped embedding is used for all the overlapped users (or items) in target domain.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(SSCDR, self).__init__(config, dataset)

        assert self.overlapped_num_items == 1 or self.overlapped_num_users == 1, \
            "SSCDR model only support user overlapped or item overlapped dataset! "
        if self.overlapped_num_users > 1:
            self.mode = 'overlap_users'
        elif self.overlapped_num_items > 1:
            self.mode = 'overlap_items'
        else:
            self.mode = 'non_overlap'
        self.phase = None
        self.dataset = dataset.source_domain_dataset.inter_feat
        # load dataset info
        self.embedding_size = config['embedding_size']
        self.lamda = config['lambda']
        self.margin = config['margin']
        self.mlp_hidden_size = config['mlp_hidden_size']
        self.mapping_layer = MLPLayers(layers=[self.embedding_size] + self.mlp_hidden_size + [self.embedding_size],
                                            activation='tanh', dropout=0, bn=False)
        if self.mode == 'overlap_users':
            self.user_interacted_items = self.build_interacted_items(dataset, mode='user')
        elif self.mode == 'overlap_items':
            self.item_interacted_users = self.build_interacted_items(dataset, mode='item')

        # define layers and loss
        self.source_user_embedding = torch.nn.Embedding(self.total_num_users, self.embedding_size)
        self.source_item_embedding = torch.nn.Embedding(self.total_num_items, self.embedding_size)

        self.target_user_embedding = torch.nn.Embedding(self.total_num_users, self.embedding_size)
        self.target_item_embedding = torch.nn.Embedding(self.total_num_items, self.embedding_size)

        with torch.no_grad():
            self.source_user_embedding.weight[self.overlapped_num_users: self.target_num_users].fill_(0)
            self.source_item_embedding.weight[self.overlapped_num_items: self.target_num_items].fill_(0)

            self.target_user_embedding.weight[self.target_num_users:].fill_(0)
            self.target_item_embedding.weight[self.target_num_items:].fill_(0)

        self.map_loss = nn.MSELoss()
        self.rec_loss = nn.TripletMarginLoss(margin=self.margin)

        # parameters initialization
        self.apply(xavier_normal_initialization)

    def build_interacted_items(self, dataset, mode='user'):
        dataset = dataset.source_domain_dataset
        if mode == 'user':
            interacted_items = [[] for _ in range(self.total_num_users)]
            for uid, iid in zip(dataset.inter_feat[dataset.uid_field].numpy(),
                                dataset.inter_feat[dataset.iid_field].numpy()):
                interacted_items[uid].append(iid)
            return interacted_items
        else:
            interacted_users = [[] for _ in range(self.total_num_items)]
            for iid, uid in zip(dataset.inter_feat[dataset.iid_field].numpy(),
                                dataset.inter_feat[dataset.uid_field].numpy()):
                interacted_users[iid].append(uid)
            return interacted_users

    def sample(self, ids, mode='user'):
        ids = ids.cpu().numpy()
        interacted = np.zeros_like(ids)
        non_interacted = np.zeros_like(ids)
        if mode =='user':
            all_candidates = list(range(self.overlapped_num_items)) + \
                             list(range(self.target_num_items, self.total_num_items))
            for index, id in enumerate(ids):
                interacted_items = self.user_interacted_items[id]
                if len(interacted_items) == 0:
                    interacted_items.append(0)
                non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                while non_interacted_id in interacted_items:
                    non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                interacted[index] = np.random.choice(interacted_items, size=1)[0]
                non_interacted[index] = non_interacted_id
        else:
            all_candidates = list(range(self.overlapped_num_users)) + \
                             list(range(self.target_num_users, self.total_num_users))
            for index, id in enumerate(ids):
                interacted_users = self.item_interacted_users[id]
                if len(interacted_users) == 0:
                    interacted_users.append(0)
                non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                while non_interacted_id in interacted_users:
                    non_interacted_id = np.random.choice(all_candidates, size=1)[0]
                interacted[index] = np.random.choice(interacted_users, size=1)[0]
                non_interacted[index] = non_interacted_id
        return torch.from_numpy(interacted).to(self.device), torch.from_numpy(non_interacted).to(self.device)

    @staticmethod
    def embedding_normalize(embeddings):
        emb_length = torch.sum(embeddings**2, dim=1, keepdim=True)
        ones = torch.ones_like(emb_length)
        norm = torch.where(emb_length > 1, emb_length, ones)
        return embeddings / norm

    @staticmethod
    def embedding_distance(emb1, emb2):
        return torch.sum((emb1-emb2)**2, dim=1)

    def set_phase(self, phase):
        self.phase = phase

    def calculate_source_loss(self, interaction):
        source_user = interaction[self.SOURCE_USER_ID]
        source_pos_item = interaction[self.SOURCE_ITEM_ID]
        source_neg_item = interaction[self.SOURCE_NEG_ITEM_ID]

        source_user_e = self.source_user_embedding(source_user)
        source_pos_item_e = self.source_item_embedding(source_pos_item)
        source_neg_item_e = self.source_item_embedding(source_neg_item)

        loss_t = self.rec_loss(self.embedding_normalize(source_user_e),
                               self.embedding_normalize(source_pos_item_e),
                               self.embedding_normalize(source_neg_item_e))
        return loss_t

    def calculate_target_loss(self, interaction):
        target_user = interaction[self.TARGET_USER_ID]
        target_pos_item = interaction[self.TARGET_ITEM_ID]
        target_neg_item = interaction[self.TARGET_NEG_ITEM_ID]

        target_user_e = self.target_user_embedding(target_user)
        target_pos_item_e = self.target_item_embedding(target_pos_item)
        target_neg_item_e = self.target_item_embedding(target_neg_item)

        loss_t = self.rec_loss(self.embedding_normalize(target_user_e),
                               self.embedding_normalize(target_pos_item_e),
                               self.embedding_normalize(target_neg_item_e))
        return loss_t

    def calculate_map_loss(self, interaction):
        idx = interaction[self.OVERLAP_ID].squeeze(1)
        if self.mode == 'overlap_users':
            source_user_e = self.source_user_embedding(idx)
            target_user_e = self.target_user_embedding(idx)
            map_e = self.mapping_layer(source_user_e)
            loss_s = self.map_loss(map_e, target_user_e)
            source_pos_item, source_neg_item = self.sample(idx, mode='user')

            map_pos_item_e = self.mapping_layer(self.source_item_embedding(source_pos_item))
            map_neg_item_e = self.mapping_layer(self.source_item_embedding(source_neg_item))
            loss_u = self.rec_loss(self.embedding_normalize(target_user_e),
                                   self.embedding_normalize(map_pos_item_e),
                                   self.embedding_normalize(map_neg_item_e))
        else:
            source_item_e = self.source_item_embedding(idx)
            target_item_e = self.target_item_embedding(idx)
            map_e = self.mapping_layer(source_item_e)
            loss_s = self.map_loss(map_e, target_item_e)
            source_pos_user, source_neg_user = self.sample(idx, mode='item')

            map_pos_user_e = self.mapping_layer(self.source_user_embedding(source_pos_user))
            map_neg_user_e = self.mapping_layer(self.source_user_embedding(source_neg_user))
            loss_u = self.rec_loss(self.embedding_normalize(target_item_e),
                                   self.embedding_normalize(map_pos_user_e),
                                   self.embedding_normalize(map_neg_user_e))
        return loss_s + self.lamda * loss_u

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
            user_e = self.embedding_normalize(self.source_user_embedding(user))
            item_e = self.embedding_normalize(self.source_item_embedding(item))
            score = -self.embedding_distance(user_e, item_e)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            user_e = self.embedding_normalize(self.target_user_embedding(user))
            item_e = self.embedding_normalize(self.target_item_embedding(item))
            score = -self.embedding_distance(user_e, item_e)
        else:
            user = interaction[self.TARGET_USER_ID]
            item = interaction[self.TARGET_ITEM_ID]
            if self.mode == 'overlap_users':
                repeat_user = user.repeat(self.embedding_size, 1).transpose(0, 1)
                user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping_layer(self.source_user_embedding(user)),
                                     self.target_user_embedding(user))
                item_e = self.target_item_embedding(item)
            else:
                user_e = self.target_user_embedding(user)
                repeat_item = item.repeat(self.embedding_size, 1).transpose(0, 1)
                item_e = torch.where(repeat_item < self.overlapped_num_items, self.mapping_layer(self.source_item_embedding(item)),
                                     self.target_item_embedding(item))
            user_e = self.embedding_normalize(user_e)
            item_e = self.embedding_normalize(item_e)
            score = -self.embedding_distance(user_e, item_e)
        return score

    def full_sort_predict(self, interaction):
        if self.phase == 'SOURCE':
            user = interaction[self.SOURCE_USER_ID]
            user_e = self.embedding_normalize(self.source_user_embedding(user))
            overlap_item_e = self.embedding_normalize(self.source_item_embedding.weight[:self.overlapped_num_items])
            source_item_e = self.embedding_normalize(self.source_item_embedding.weight[self.target_num_items:])
            all_item_e = torch.cat([overlap_item_e, source_item_e], dim=0)
        elif self.phase == 'TARGET':
            user = interaction[self.TARGET_USER_ID]
            user_e = self.embedding_normalize(self.target_user_embedding(user))
            all_item_e = self.embedding_normalize(self.target_item_embedding.weight[:self.target_num_items])
        else:
            user = interaction[self.TARGET_USER_ID]
            if self.mode == 'overlap_users':
                repeat_user = user.repeat(self.embedding_size, 1).transpose(0, 1)
                user_e = torch.where(repeat_user < self.overlapped_num_users, self.mapping_layer(self.source_user_embedding(user)),
                                     self.target_user_embedding(user))
                all_item_e = self.target_item_embedding.weight[:self.target_num_items]
            else:
                user_e = self.target_user_embedding(user)
                overlap_item_e = self.mapping_layer(self.source_item_embedding.weight[:self.overlapped_num_items])
                target_item_e = self.target_item_embedding.weight[self.overlapped_num_items:self.target_num_items]
                all_item_e = torch.cat([overlap_item_e, target_item_e], dim=0)
            user_e = self.embedding_normalize(user_e)
            all_item_e = self.embedding_normalize(all_item_e)

        num_batch_user, emb_dim = user_e.size()
        num_all_item, _ = all_item_e.size()
        dist = -2 * torch.matmul(user_e, all_item_e.permute(1, 0))
        dist += torch.sum(user_e ** 2, -1).view(num_batch_user, 1)
        dist += torch.sum(all_item_e ** 2, -1).view(1, num_all_item)
        return -dist.view(-1)

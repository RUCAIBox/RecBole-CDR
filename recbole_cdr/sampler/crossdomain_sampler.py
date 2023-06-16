# -*- coding: utf-8 -*-
# @Time   : 2022/3/12
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# @File   : crossdomain_sampler.py

"""
recbole_cdr.sampler
########################
"""

import copy

import numpy as np
from numpy.random import sample
import torch
from collections import Counter


class AbstractSampler(object):
    """:class:`AbstractSampler` is a abstract class, all sampler should inherit from it. This sampler supports returning
    a certain number of random value_ids according to the input key_id, and it also supports to prohibit
    certain key-value pairs by setting used_ids.

    Args:
        distribution (str): The string of distribution, which is used for subclass.

    Attributes:
        used_ids (numpy.ndarray): The result of :meth:`get_used_ids`.
    """

    def __init__(self, distribution):
        self.distribution = ''
        self.set_distribution(distribution)
        self.used_ids = self.get_used_ids()

    def set_distribution(self, distribution):
        """Set the distribution of sampler.

        Args:
            distribution (str): Distribution of the negative items.
        """
        self.distribution = distribution
        if distribution == 'popularity':
            self._build_alias_table()

    def _uni_sampling(self, sample_num):
        """Sample [sample_num] items in the uniform distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """
        raise NotImplementedError('Method [_uni_sampling] should be implemented')

    def _get_candidates_list(self):
        """Get sample candidates list for _pop_sampling()

        Returns:
            candidates_list (list): a list of candidates id.
        """
        raise NotImplementedError('Method [_get_candidates_list] should be implemented')

    def _build_alias_table(self):
        """Build alias table for popularity_biased sampling.
        """
        candidates_list = self._get_candidates_list()
        self.prob = dict(Counter(candidates_list))
        self.alias = self.prob.copy()
        large_q = []
        small_q = []

        for i in self.prob:
            self.alias[i] = -1
            self.prob[i] = self.prob[i] / len(candidates_list) * len(self.prob)
            if self.prob[i] > 1:
                large_q.append(i)
            elif self.prob[i] < 1:
                small_q.append(i)

        while len(large_q) != 0 and len(small_q) != 0:
            l = large_q.pop(0)
            s = small_q.pop(0)
            self.alias[s] = l
            self.prob[l] = self.prob[l] - (1 - self.prob[s])
            if self.prob[l] < 1:
                small_q.append(l)
            elif self.prob[l] > 1:
                large_q.append(l)

    def _pop_sampling(self, sample_num):
        """Sample [sample_num] items in the popularity-biased distribution.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples.
        """

        keys = list(self.prob.keys())
        random_index_list = np.random.randint(0, len(keys), sample_num)
        random_prob_list = np.random.random(sample_num)
        final_random_list = []

        for idx, prob in zip(random_index_list, random_prob_list):
            if self.prob[keys[idx]] > prob:
                final_random_list.append(keys[idx])
            else:
                final_random_list.append(self.alias[keys[idx]])

        return np.array(final_random_list)

    def sampling(self, sample_num):
        """Sampling [sample_num] item_ids.

        Args:
            sample_num (int): the number of samples.

        Returns:
            sample_list (np.array): a list of samples and the len is [sample_num].
        """
        if self.distribution == 'uniform':
            return self._uni_sampling(sample_num)
        elif self.distribution == 'popularity':
            return self._pop_sampling(sample_num)
        else:
            raise NotImplementedError(f'The sampling distribution [{self.distribution}] is not implemented.')

    def get_used_ids(self):
        """
        Returns:
            numpy.ndarray: Used ids. Index is key_id, and element is a set of value_ids.
        """
        raise NotImplementedError('Method [get_used_ids] should be implemented')

    def sample_by_key_ids(self, key_ids, num):
        """Sampling by key_ids.

        Args:
            key_ids (numpy.ndarray or list): Input key_ids.
            num (int): Number of sampled value_ids for each key_id.

        Returns:
            torch.tensor: Sampled value_ids.
            value_ids[0], value_ids[len(key_ids)], value_ids[len(key_ids) * 2], ..., value_id[len(key_ids) * (num - 1)]
            is sampled for key_ids[0];
            value_ids[1], value_ids[len(key_ids) + 1], value_ids[len(key_ids) * 2 + 1], ...,
            value_id[len(key_ids) * (num - 1) + 1] is sampled for key_ids[1]; ...; and so on.
        """
        key_ids = np.array(key_ids)
        key_num = len(key_ids)
        total_num = key_num * num
        if (key_ids == key_ids[0]).all():
            key_id = key_ids[0]
            used = np.array(list(self.used_ids[key_id]))
            value_ids = self.sampling(total_num)
            check_list = np.arange(total_num)[np.isin(value_ids, used)]
            while len(check_list) > 0:
                value_ids[check_list] = value = self.sampling(len(check_list))
                mask = np.isin(value, used)
                check_list = check_list[mask]
        else:
            value_ids = np.zeros(total_num, dtype=np.int64)
            check_list = np.arange(total_num)
            key_ids = np.tile(key_ids, num)
            while len(check_list) > 0:
                value_ids[check_list] = self.sampling(len(check_list))
                check_list = np.array([
                    i for i, used, v in zip(check_list, self.used_ids[key_ids[check_list]], value_ids[check_list])
                    if v in used
                ])
        return torch.tensor(value_ids)


class CrossDomainSourceSampler(AbstractSampler):
    """:class:`CrossDomainSourceSampler` is used to sample negative items for source domain dataset.

        Args:
            phases (list of str): The list that contains the phases for sampling.
            dataset (Dataset): The source domain dataset, which contains data interaction in source domain.
            distribution (str, optional): Distribution of the negative entities. Defaults to 'uniform'.
        """

    def __init__(self, phases, dataset, built_datasets, distribution='uniform'):
        if not isinstance(phases, list):
            phases = [phases]

        self.phases = phases
        self.dataset = dataset.source_domain_dataset
        self.datasets = built_datasets

        self.uid_field = self.dataset.uid_field
        self.iid_field = self.dataset.iid_field

        self.overlapped_item_num = dataset.num_overlap_item
        self.overlapped_user_num = dataset.num_overlap_user

        self.source_only_item_num = dataset.num_source_only_item
        self.source_only_user_num = dataset.num_source_only_user

        self.target_only_item_num = dataset.num_target_only_item
        self.target_only_user_num = dataset.num_target_only_user

        self.total_user_num = dataset.num_total_user
        self.total_item_num = dataset.num_total_item

        self.item_num = self.overlapped_item_num + self.source_only_item_num

        self.item_id_list = np.array(list(range(1, self.overlapped_item_num)) + \
                            list(range(self.overlapped_item_num + self.target_only_item_num, self.total_item_num)))

        self.user_id_list = np.array(list(range(1, self.overlapped_user_num)) + \
                                     list(range(self.overlapped_user_num + self.target_only_user_num,
                                                self.total_user_num)))
        super().__init__(distribution=distribution)

    def _uni_sampling(self, sample_num):
        return np.random.choice(self.item_id_list, size=sample_num, replace=True)

    def _get_candidates_list(self):
        candidates_list = []
        for dataset in self.datasets:
            candidates_list.extend(dataset.inter_feat[self.iid_field].numpy())
        return candidates_list

    def get_used_ids(self):
        """
        Returns:
            dict: Used item_ids is the same as positive item_ids.
            Key is phase, and value is a numpy.ndarray which index is user_id, and element is a set of item_ids.
        """
        used_item_id = dict()
        last = [set() for _ in range(self.total_user_num)]
        for phase in self.phases:
            cur = np.array([set(s) for s in last])
            for uid, iid in zip(self.dataset.inter_feat[self.uid_field].numpy(), self.dataset.inter_feat[self.iid_field].numpy()):
                cur[uid].add(iid)
            last = used_item_id[phase] = cur

        for used_item_set in used_item_id[self.phases[-1]]:
            if len(used_item_set) + 1 == self.item_num:  # [pad] is a item.
                raise ValueError(
                    'Some users have interacted with all items, '
                    'which we can not sample negative items for them. '
                    'Please set `user_inter_num_interval` to filter those users.'
                )
        return used_item_id

    def set_phase(self, phase):
        """Get the sampler of corresponding phase.

        Args:
            phase (str): The phase of new sampler.

        Returns:
            Sampler: the copy of this sampler, :attr:`phase` is set the same as input phase, and :attr:`used_ids`
            is set to the value of corresponding phase.
        """
        if phase not in self.phases:
            raise ValueError(f'Phase [{phase}] not exist.')
        new_sampler = copy.copy(self)
        new_sampler.phase = phase
        new_sampler.used_ids = new_sampler.used_ids[phase]
        return new_sampler

    def sample_by_user_ids(self, user_ids, item_ids, num):
        """Sampling by user_ids.

        Args:
            user_ids (numpy.ndarray or list): Input user_ids.
            item_ids (numpy.ndarray or list): Input item_ids.
            num (int): Number of sampled item_ids for each user_id.

        Returns:
            torch.tensor: Sampled item_ids.
            item_ids[0], item_ids[len(user_ids)], item_ids[len(user_ids) * 2], ..., item_id[len(user_ids) * (num - 1)]
            is sampled for user_ids[0];
            item_ids[1], item_ids[len(user_ids) + 1], item_ids[len(user_ids) * 2 + 1], ...,
            item_id[len(user_ids) * (num - 1) + 1] is sampled for user_ids[1]; ...; and so on.
        """
        try:
            return self.sample_by_key_ids(user_ids, num)
        except IndexError:
            for user_id in user_ids:
                if user_id not in self.user_id_list:
                    raise ValueError(f'user_id [{user_id}] not exist.')

# @Time   : 2022/3/10
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   : 2022/4/9
# @Author : Gaowei Zhang
# @email  : 1462034631@qq.com

"""
recbole_cdr.data.dataloader
################################################
"""

from logging import getLogger
import numpy as np
import torch

from recbole.data.interaction import Interaction
from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.general_dataloader import TrainDataLoader, FullSortEvalDataLoader

from recbole_cdr.utils import CrossDomainDataLoaderState


class OverlapDataloader(AbstractDataLoader):
    """:class:`OverlapDataloader` is a dataloader for training algorithms with only overlapped users or items.

    Args:
        config (Config): The config of dataloader.
        dataset (Dataset): The dataset of dataloader.
        sampler (Sampler): The sampler of dataloader in source domain.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """
    def __init__(self, config, dataset, sampler=None, shuffle=False):
        super().__init__(config, dataset, sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        batch_size = self.config['overlap_batch_size']
        self.step = batch_size
        self.set_batch_size(batch_size)

    @property
    def pr_end(self):
        return len(self.dataset)

    def _shuffle(self):
        self.dataset.shuffle()

    def _next_batch_data(self):
        cur_data = self.dataset[self.pr:self.pr + self.step]
        self.pr += self.step
        return cur_data


class CrossDomainDataloader(AbstractDataLoader):
    """:class:`CrossDomainDataLoader` is a dataloader for training Cross domain algorithms.

    Args:
        config (Config): The config of dataloader.
        source_dataset (Dataset): The dataset of dataloader in source domain.
        source_sampler (Sampler): The sampler of dataloader in source domain.
        target_dataset (Dataset): The dataset of dataloader in target domain.
        target_sampler (Sampler): The sampler of dataloader in target domain.
        shuffle (bool, optional): Whether the dataloader will be shuffled after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, source_dataset, source_sampler, target_dataset, target_sampler,
                 shuffle=False):
        config.update(config['source_domain'])
        config['LABEL_FIELD'] = source_dataset.label_field
        config['NEG_PREFIX'] = source_dataset.neg_prefix
        self.source_dataloader = TrainDataLoader(config, source_dataset, source_sampler, shuffle=shuffle)
        config.update(config['target_domain'])
        config['LABEL_FIELD'] = target_dataset.label_field
        config['NEG_PREFIX'] = target_dataset.neg_prefix
        self.target_dataloader = TrainDataLoader(config, target_dataset, target_sampler, shuffle=shuffle)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.state = CrossDomainDataLoaderState.BOTH

        super().__init__(config, dataset, target_sampler, shuffle=shuffle)
        self.dataset.target_domain_dataset = target_dataset
        self.overlap_dataset = self.dataset.overlap_dataset
        self.overlap_dataloader = OverlapDataloader(config, self.overlap_dataset, sampler=None, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        pass

    def reinit_pr_after_map(self):
        self.source_dataloader.pr = 0
        self.target_dataloader.pr = 0

    def update_config(self, config):
        self.source_dataloader.update_config(config)
        self.target_dataloader.update_config(config)
        self.overlap_dataset.update_config(config)

    def __iter__(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.__iter__()
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return self.target_dataloader.__iter__()
        elif self.state == CrossDomainDataLoaderState.BOTH:
            self.source_dataloader.__iter__()
            self.target_dataloader.__iter__()
            return self
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return self.overlap_dataloader.__iter__()

    def _shuffle(self):
        pass

    def __next__(self):
        if self.state == CrossDomainDataLoaderState.SOURCE and self.source_dataloader.pr >= self.source_dataloader.pr_end:
            self.target_dataloader.pr = 0
            self.source_dataloader.pr = 0
            raise StopIteration()
        if self.state == CrossDomainDataLoaderState.TARGET or self.state == CrossDomainDataLoaderState.BOTH:
            if self.target_dataloader.pr >= self.target_dataloader.pr_end:
                self.target_dataloader.pr = 0
                self.source_dataloader.pr = 0
                raise StopIteration()
        if self.state == CrossDomainDataLoaderState.OVERLAP and self.overlap_dataloader.pr >= self.overlap_dataloader.pr_end:
            self.overlap_dataloader.pr = 0
            raise StopIteration()
        return self._next_batch_data()

    def __len__(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return len(self.source_dataloader)
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return len(self.target_dataloader)
        elif self.state == CrossDomainDataLoaderState.BOTH:
            return len(self.target_dataloader)
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return len(self.overlap_dataloader)

    @property
    def pr_end(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.pr_end
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return self.overlap_dataloader.pr_end
        else:
            return self.target_dataloader.pr_end

    def _next_batch_data(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.__next__()
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return self.target_dataloader.__next__()
        elif self.state == CrossDomainDataLoaderState.OVERLAP:
            return self.overlap_dataloader.__next__()
        else:
            try:
                source_data = self.source_dataloader.__next__()
            except StopIteration:
                source_data = self.source_dataloader.__next__()
            target_data = self.target_dataloader.__next__()
            target_data.update(source_data)
            return target_data

    def set_mode(self, state):
        """Set the mode of :class:`CrossDomainDataloaderDataLoader`, it can be set to three states:

            - CrossDomainDataLoaderState.BOTH
            - CrossDomainDataLoaderState.SOURCE
            - CrossDomainDataLoaderState.TARGET

        The state of :class:`CrossDomainDataloaderDataLoader` would affect the result of _next_batch_data().

        Args:
            state (CrossDomainDataloaderState): the state of :class:`CrossDomainDataloaderDataLoader`.
        """
        if state not in set(CrossDomainDataLoaderState):
            raise NotImplementedError(f'Cross Domain data loader has no state named [{state}].')
        if self.source_dataloader.pr != 0 or self.target_dataloader.pr != 0:
            raise PermissionError('Cannot change dataloader\'s state within an epoch')
        self.state = state

    def get_model(self, model):
        """Let the dataloader get the model, used for dynamic sampling.
        """
        self.source_dataloader.get_model(model)
        self.target_dataloader.get_model(model)


class CrossDomainFullSortEvalDataLoader(FullSortEvalDataLoader):
    """:class:`CrossdomainFullSortEvalDataLoader` is a dataloader for full-sort evaluation. In order to speed up calculation,
    this dataloader would only return then user part of interactions, positive items and used items.
    It would not return negative items.

    Args:
        config (Config): The config of dataloader.
        dataset (CrossDomainDataset): The dataset from both domain.
        source_dataset(CrossDomainSingleDataset): The dataset that only from source domain.
        sampler (Sampler): The sampler of dataloader.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, source_dataset, sampler, shuffle=False):
        self.uid_field = source_dataset.uid_field
        self.iid_field = source_dataset.iid_field
        self.is_sequential = False

        user_num = dataset.num_total_user
        self.overlap_item_num = dataset.num_overlap_item
        self.revoke_item_num = dataset.num_target_only_item
        self.uid_list = []
        self.uid2items_num = np.zeros(user_num, dtype=np.int64)
        self.uid2positive_item = np.array([None] * user_num)
        self.uid2history_item = np.array([None] * user_num)

        source_dataset.sort(by=self.uid_field, ascending=True)
        last_uid = None
        positive_item = set()
        uid2used_item = sampler.used_ids
        for uid, iid in zip(source_dataset.inter_feat[self.uid_field].numpy(),
                            source_dataset.inter_feat[self.iid_field].numpy()):
            if uid != last_uid:
                self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
                last_uid = uid
                self.uid_list.append(uid)
                positive_item = set()
            positive_item.add(iid)
        self._set_user_property(last_uid, uid2used_item[last_uid], positive_item)
        self.uid_list = torch.tensor(self.uid_list, dtype=torch.int64)
        self.user_df = source_dataset.join(Interaction({self.uid_field: self.uid_list}))

        self.config = config
        self.logger = getLogger()
        self.dataset = source_dataset
        self.sampler = sampler
        self.batch_size = self.step = self.model = None
        self.shuffle = shuffle
        self.pr = 0
        self._init_batch_size_and_step()

    def _set_user_property(self, uid, used_item, positive_item):
        if uid is None:
            return
        history_item = used_item - positive_item
        revoke_map_pos_item = [iid if iid < self.overlap_item_num else iid - self.revoke_item_num for iid in list(positive_item)]
        revoke_map_his_item = [iid if iid < self.overlap_item_num else iid - self.revoke_item_num for iid in list(history_item)]
        self.uid2positive_item[uid] = torch.tensor(revoke_map_pos_item, dtype=torch.int64)
        self.uid2items_num[uid] = len(positive_item)
        self.uid2history_item[uid] = torch.tensor(revoke_map_his_item, dtype=torch.int64)
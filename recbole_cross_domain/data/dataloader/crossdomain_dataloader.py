# @Time   : 2022/3/10
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole_cross_domain.data.dataloader.crossdomain_dataloader
################################################
"""

from recbole.data.dataloader.abstract_dataloader import AbstractDataLoader
from recbole.data.dataloader.general_dataloader import TrainDataLoader

from recbole_cross_domain.utils import CrossDomainDataLoaderState


class CrossDomainDataloader(AbstractDataLoader):
    """:class:`CrossDomainDataLoader` is a dataloader for training Cross domain algorithms.

    Args:
        config (Config): The config of dataloader.
        source_dataset (Dataset): The dataset of dataloader in source domain.
        source_sampler (Sampler): The sampler of dataloader in source domain.
        target_dataset (Dataset): The dataset of dataloader in target domain.
        target_sampler (Sampler): The sampler of dataloader in target domain.
        shuffle (bool, optional): Whether the dataloader will be shuffle after a round. Defaults to ``False``.
    """

    def __init__(self, config, dataset, source_dataset, source_sampler, target_dataset, target_sampler,
                 shuffle=False):
        config.update(config['source_domain'])
        config['LABEL_FIELD'] = source_dataset.label_field
        self.source_dataloader = TrainDataLoader(config, source_dataset, source_sampler, shuffle=shuffle)
        config.update(config['target_domain'])
        config['LABEL_FIELD'] = target_dataset.label_field
        self.target_dataloader = TrainDataLoader(config, target_dataset, target_sampler, shuffle=shuffle)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.state = CrossDomainDataLoaderState.BOTH

        super().__init__(config, dataset, target_sampler, shuffle=shuffle)

    def _init_batch_size_and_step(self):
        pass

    def update_config(self, config):
        self.source_dataloader.update_config(config)
        self.target_dataloader.update_config(config)

    def __iter__(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.__iter__()
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return self.target_dataloader.__iter__()
        elif self.state == CrossDomainDataLoaderState.BOTH:
            self.source_dataloader.__iter__()
            self.target_dataloader.__iter__()
            return self

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
        return self._next_batch_data()

    def __len__(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return len(self.source_dataloader)
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return len(self.target_dataloader)
        elif self.state == CrossDomainDataLoaderState.BOTH:
            return len(self.target_dataloader)

    @property
    def pr_end(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.pr_end
        else:
            return self.target_dataloader.pr_end

    def _next_batch_data(self):
        if self.state == CrossDomainDataLoaderState.SOURCE:
            return self.source_dataloader.__next__()
        elif self.state == CrossDomainDataLoaderState.TARGET:
            return self.target_dataloader.__next__()
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

# @Time   : 2022/3/12
# @Author : zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   : 2022/4/9
# @Author : Gaowei Zhang
# @email  : 1462034631@qq.com

r"""
recbole_cdr.trainer.trainer
################################
"""

import numpy as np
from recbole.trainer import Trainer
from recbole_cdr.utils import CrossDomainDataLoaderState


class CrossDomainTrainer(Trainer):
    r"""

    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)


class EMCDRTrainer(Trainer):
    r"""

    """

    def __init__(self, config, model):
        super(EMCDRTrainer, self).__init__(config, model)
        self.source_train_step = config['source_train_step']
        self.target_train_step = config['target_train_step']
        self.phase_list = [CrossDomainDataLoaderState.SOURCE, CrossDomainDataLoaderState.TARGET,
                           CrossDomainDataLoaderState.MAP, CrossDomainDataLoaderState.BOTH]
        self.phase_pr = 0
        self.phase = self.phase_list[self.phase_pr]

    def _reinit(self):
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.train_loss_dict = dict()
        self.phase_pr += 1
        self.phase = self.phase_list[self.phase_pr]

    '''def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        if self.phase == CrossDomainDataLoaderState.SOURCE:
            return super()._train_epoch(
                train_data, epoch_idx, loss_func=self.model.calculate_source_loss, show_progress=show_progress
            )
        elif self.phase in [CrossDomainDataLoaderState.TARGET, CrossDomainDataLoaderState.BOTH]:
            return super()._train_epoch(
                train_data, epoch_idx, loss_func=self.model.calculate_target_loss, show_progress=show_progress
            )
        elif self.phase == CrossDomainDataLoaderState.MAP:
            return super()._train_epoch(
                train_data, epoch_idx, loss_func=self.model.calculate_map_loss, show_progress=show_progress
            )
        return None'''

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        train_data.set_mode(CrossDomainDataLoaderState.SOURCE)
        self.model.set_phase('source')
        super().fit(train_data, None, verbose, saved, show_progress, callback_fn)

        self._reinit()
        train_data.set_mode(CrossDomainDataLoaderState.TARGET)
        self.model.set_phase('target')
        super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self._reinit()
        train_data.set_mode(CrossDomainDataLoaderState.MAP)
        train_data.reinit_pr_after_map()
        self.model.set_phase('map')
        super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        return self.best_valid_score, self.best_valid_result
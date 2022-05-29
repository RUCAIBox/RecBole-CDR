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
from recbole_cdr.utils import train_mode2state


class CrossDomainTrainer(Trainer):
    r"""Trainer for training cross-domain models. It contains four training mode: SOURCE, TARGET, BOTH, OVERLAP
    which can be set by the parameter of `train_epochs`
    """

    def __init__(self, config, model):
        super(CrossDomainTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if self.split_valid_flag and valid_data is not None:
                source_valid_data, target_valid_data = valid_data
                if scheme == 'SOURCE':
                    super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
            else:
                super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result


class DCDCSRTrainer(Trainer):
    r"""Trainer for training DCDCSR models."""

    def __init__(self, config, model):
        super(DCDCSRTrainer, self).__init__(config, model)
        self.train_modes = config['train_modes']
        self.train_epochs = config['epoch_num']
        self.split_valid_flag = config['source_split']

    def _reinit(self, phase):
        """Reset the parameters when start a new training phase.
        """
        self.start_epoch = 0
        self.cur_step = 0
        self.best_valid_score = -np.inf if self.valid_metric_bigger else np.inf
        self.best_valid_result = None
        self.item_tensor = None
        self.tot_item_num = None
        self.train_loss_dict = dict()
        self.epochs = int(self.train_epochs[phase])
        self.eval_step = min(self.config['eval_step'], self.epochs)

    def fit(self, train_data, valid_data=None, verbose=True, saved=True, show_progress=False, callback_fn=None):
        r"""Train the model based on the train data and the valid data.

            Args:
                train_data (DataLoader): the train data
                valid_data (DataLoader, optional): the valid data, default: None.
                                                    If it's None, the early_stopping is invalid.
                verbose (bool, optional): whether to write training and evaluation information to logger, default: True
                saved (bool, optional): whether to save the model parameters, default: True
                show_progress (bool): Show the progress of training epoch and evaluate epoch. Defaults to ``False``.
                callback_fn (callable): Optional callback function executed at end of epoch.
                                        Includes (epoch_idx, valid_score) input arguments.

            Returns:
                    (float, dict): best valid score and best valid result. If valid_data is None, it returns (-1, None)
        """
        for phase in range(len(self.train_modes)):
            self._reinit(phase)
            scheme = self.train_modes[phase]
            self.logger.info("Start training with {} mode".format(scheme))
            state = train_mode2state[scheme]
            train_data.set_mode(state)
            self.model.set_phase(scheme)
            if scheme == 'BOTH':
                super().fit(train_data, None, verbose, saved, show_progress, callback_fn)
            else:
                if self.split_valid_flag and valid_data is not None:
                    source_valid_data, target_valid_data = valid_data
                    if scheme == 'SOURCE':
                        super().fit(train_data, source_valid_data, verbose, saved, show_progress, callback_fn)
                    else:
                        super().fit(train_data, target_valid_data, verbose, saved, show_progress, callback_fn)
                else:
                    super().fit(train_data, valid_data, verbose, saved, show_progress, callback_fn)

        self.model.set_phase('OVERLAP')
        return self.best_valid_score, self.best_valid_result

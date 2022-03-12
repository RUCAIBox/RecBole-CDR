# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole_cross_domain.data.utils
########################
"""

import importlib
import os
import pickle

from recbole.data.dataloader import NegSampleEvalDataLoader, FullSortEvalDataLoader
from recbole.data.utils import load_split_dataloaders, save_split_dataloaders, create_samplers, getLogger
from recbole.utils import set_color
from recbole.utils.argument_list import dataset_arguments

from recbole_cross_domain.data.dataloader import *
from recbole_cross_domain.sampler import CrsssDomainSourceSampler
from recbole_cross_domain.utils import ModelType


def create_dataset(config):
    """Create cross domain dataset.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module('recbole_cross_domain.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    else:
        model_type = config['MODEL_TYPE']
        type2class = {
            ModelType.CROSSDOMAIN: 'CrossDomainDataset',
        }
        dataset_class = getattr(dataset_module, type2class[model_type])

    default_file = os.path.join(config['checkpoint_dir'], f'{config["dataset"]}-{dataset_class.__name__}.pth')
    file = config['dataset_save_path'] or default_file
    if os.path.exists(file):
        with open(file, 'rb') as f:
            dataset = pickle.load(f)
        dataset_args_unchanged = True
        for arg in dataset_arguments + ['seed', 'repeatable']:
            if config[arg] != dataset.config[arg]:
                dataset_args_unchanged = False
                break
        if dataset_args_unchanged:
            logger = getLogger()
            logger.info(set_color('Load filtered dataset from', 'pink') + f': [{file}]')
            return dataset

    dataset = dataset_class(config)
    if config['save_dataset']:
        dataset.save()
    return dataset


def data_preparation(config, dataset):
    """Split the dataset by :attr:`config['eval_args']` and create training, validation and test dataloader.

    Note:
        If we can load split dataloaders by :meth:`load_split_dataloaders`, we will not create new split dataloaders.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.

    Returns:
        tuple:
            - train_data (AbstractDataLoader): The dataloader for training.
            - valid_data (AbstractDataLoader): The dataloader for validation.
            - test_data (AbstractDataLoader): The dataloader for testing.
    """
    dataloaders = load_split_dataloaders(config)
    if dataloaders is not None:
        train_data, valid_data, test_data = dataloaders
    else:
        built_datasets = dataset.build()

        source_train_dataset, target_train_dataset, valid_dataset, test_dataset = built_datasets
        target_train_sampler, valid_sampler, test_sampler = \
            create_samplers(config, dataset.target_domain_dataset, built_datasets[1:])

        source_train_sampler = CrsssDomainSourceSampler(dataset, config['train_neg_sample_args']['distribution'])

        train_data = get_dataloader(config, 'train')(config, dataset, source_train_dataset, source_train_sampler,
                                                     target_train_dataset, target_train_sampler, shuffle=True)

        valid_data = get_dataloader(config, 'evaluation')(config, valid_dataset, valid_sampler, shuffle=False)
        test_data = get_dataloader(config, 'evaluation')(config, test_dataset, test_sampler, shuffle=False)
        if config['save_dataloaders']:
            save_split_dataloaders(config, dataloaders=(train_data, valid_data, test_data))

    logger = getLogger()
    logger.info(
        set_color('[Training]: ', 'pink') + set_color('train_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["train_batch_size"]}]', 'yellow') + set_color(' negative sampling', 'cyan') + ': ' +
        set_color(f'[{config["neg_sampling"]}]', 'yellow')
    )
    logger.info(
        set_color('[Evaluation]: ', 'pink') + set_color('eval_batch_size', 'cyan') + ' = ' +
        set_color(f'[{config["eval_batch_size"]}]', 'yellow') + set_color(' eval_args', 'cyan') + ': ' +
        set_color(f'[{config["eval_args"]}]', 'yellow')
    )
    return train_data, valid_data, test_data


def get_dataloader(config, phase):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type == ModelType.CROSSDOMAIN:
            return CrossDomainDataloader
    else:
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader

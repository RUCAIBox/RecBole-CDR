# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   : 2022/4/9
# @Author : Gaowei Zhang
# @email  : 1462034631@qq.com

"""
recbole_cdr.data.utils
########################
"""

import importlib
import os
import pickle

from recbole.data.dataloader import NegSampleEvalDataLoader
from recbole.data.utils import load_split_dataloaders, save_split_dataloaders, create_samplers
from recbole.utils import set_color
from recbole.utils.argument_list import dataset_arguments

from recbole_cdr.data.dataloader import *
from recbole_cdr.sampler import CrossDomainSourceSampler
from recbole_cdr.utils import ModelType


def create_dataset(config):
    """Create cross domain dataset.
    If :attr:`config['dataset_save_path']` file exists and
    its :attr:`config` of dataset is equal to current :attr:`config` of dataset.
    It will return the saved dataset in :attr:`config['dataset_save_path']`.

    Args:
        config (CDRConfig): An instance object of Config, used to record parameter information.

    Returns:
        Dataset: Constructed dataset.
    """
    dataset_module = importlib.import_module('recbole_cdr.data.dataset')
    if hasattr(dataset_module, config['model'] + 'Dataset'):
        dataset_class = getattr(dataset_module, config['model'] + 'Dataset')
    else:
        model_type = config['MODEL_TYPE']
        type2class = {
            ModelType.CROSSDOMAIN: 'CrossDomainDataset'
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
        config (CDRConfig): An instance object of Config, used to record parameter information.
        dataset (CrossDomainDataset): An instance object of Dataset, which contains all interaction records.

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

        source_train_dataset, source_valid_dataset, target_train_dataset, \
            target_valid_dataset, target_test_dataset = built_datasets

        target_train_sampler, target_valid_sampler, target_test_sampler = \
            create_samplers(config, dataset.target_domain_dataset, built_datasets[2:])

        if source_valid_dataset is not None:
            source_train_sampler, source_valid_sampler = create_source_samplers(config, dataset, built_datasets[:2])
            source_valid_data = get_dataloader(config, 'evaluation', 'source')(config, dataset, source_valid_dataset, source_valid_sampler, shuffle=False)
            target_valid_data = get_dataloader(config, 'evaluation', 'target')(config, target_valid_dataset, target_valid_sampler, shuffle=False)

            valid_data = (source_valid_data, target_valid_data)
        else:
            source_train_sampler = CrossDomainSourceSampler('train', dataset, config['train_neg_sample_args']['distribution']).set_phase('train')
            valid_data = get_dataloader(config, 'evaluation', 'target')(config, target_valid_dataset, target_valid_sampler, shuffle=False)

        train_data = get_dataloader(config, 'train', 'target')(config, dataset, source_train_dataset, source_train_sampler,
                                                           target_train_dataset, target_train_sampler, shuffle=True)

        test_data = get_dataloader(config, 'evaluation', 'target')(config, target_test_dataset, target_test_sampler, shuffle=False)

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


def get_dataloader(config, phase, domain='target'):
    """Return a dataloader class according to :attr:`config` and :attr:`phase`.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        phase (str): The stage of dataloader. It can only take two values: 'train' or 'evaluation'.
        domain (str): The domain of Evaldataloader. It can only take two values: 'source' or 'target'.

    Returns:
        type: The dataloader class that meets the requirements in :attr:`config` and :attr:`phase`.
    """
    model_type = config['MODEL_TYPE']
    if phase == 'train':
        if model_type == ModelType.CROSSDOMAIN:
            return CrossDomainDataloader
    else:
        if domain == 'source':
            return CrossDomainFullSortEvalDataLoader
        eval_strategy = config['eval_neg_sample_args']['strategy']
        if eval_strategy in {'none', 'by'}:
            return NegSampleEvalDataLoader
        elif eval_strategy == 'full':
            return FullSortEvalDataLoader


def create_source_samplers(config, dataset, built_datasets):
    """Create sampler for training, validation and testing.

    Args:
        config (Config): An instance object of Config, used to record parameter information.
        dataset (Dataset): An instance object of Dataset, which contains all interaction records.
        built_datasets (list of Dataset): A list of split Dataset, which contains dataset for
            training, validation and testing.

    Returns:
        tuple:
            - train_sampler (AbstractSampler): The sampler for training.
            - valid_sampler (AbstractSampler): The sampler for validation.
    """
    phases = ['train', 'valid']
    train_neg_sample_args = config['train_neg_sample_args']
    eval_neg_sample_args = config['eval_neg_sample_args']

    sampler = CrossDomainSourceSampler(phases, dataset, built_datasets, train_neg_sample_args['distribution'])
    train_sampler = sampler.set_phase('train')

    sampler = CrossDomainSourceSampler(phases, dataset, built_datasets, eval_neg_sample_args['distribution'])
    valid_sampler = sampler.set_phase('valid')

    return train_sampler, valid_sampler

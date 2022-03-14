# @Time   : 2022/3/8
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole_cdr.data.crossdomain_dataset
##########################
"""

import os
import copy
from collections import ChainMap

import numpy as np
import pandas as pd
from logging import getLogger

from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource, FeatureType, set_color


class CrossDomainSingleDataset(Dataset):
    def __init__(self, config, domain='source'):
        self.domain = domain
        super().__init__(config)

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
        - Remap ID
        - Missing value imputation
        - Normalization
        - Preloading weights initialization
        """
        self.feat_name_list = self._build_feat_name_list()
        if self.benchmark_filename_list is None:
            self._data_filtering()

    def _get_preset(self):
        """Initialization useful inside attributes.
        """
        self.dataset_path = self.config['data_path']

        self.field2type = {}
        self.field2source = {}
        self.field2id_token = {}
        self.field2token_id = {}
        self.field2seqlen = self.config['seq_len'] or {}
        self.alias = {}
        self._preloaded_weight = {}
        self.benchmark_filename_list = self.config['benchmark_filename']
        self.neg_prefix = self.config['NEG_PREFIX']

    def _rename_columns(self):
        """Add the prefix of column name as source or target.
        """
        if self.uid_field:
            self.uid_field = "{}_{}".format(self.domain, self.config['USER_ID_FIELD'])
        if self.iid_field:
            self.iid_field = "{}_{}".format(self.domain, self.config['ITEM_ID_FIELD'])
        if self.label_field:
            self.label_field = "{}_{}".format(self.domain, self.config['LABEL_FIELD'])
        if self.time_field:
            self.time_field = "{}_{}".format(self.domain, self.config['TIME_FIELD'])
        if self.inter_feat is not None:
            self.inter_feat.columns = ["{}_{}".format(self.domain, col) for col in self.inter_feat.columns]
        if self.user_feat is not None:
            self.user_feat.columns = ["{}_{}".format(self.domain, col) for col in self.user_feat.columns]
        if self.item_feat is not None:
            self.item_feat.columns = ["{}_{}".format(self.domain, col) for col in self.item_feat.columns]

        dict_list = [self.field2type, self.field2source, self.field2id_token, self.field2token_id, self.field2seqlen]

        for d in dict_list:
            keys = list(d.keys())
            for key in keys:
                new_key = "{}_{}".format(self.domain, key)
                d[new_key] = d[key]
                del d[key]

    def remap_user_item_id(self, uid_remap_dict, iid_remap_dict):

        for alias in self.alias.values():
            if uid_remap_dict and self.uid_field in alias:
                self.logger.debug(set_color('map_source_user_field_to_target', 'blue'))
                self._remap_fields(alias, uid_remap_dict)
            if iid_remap_dict and self.iid_field in alias:
                self.logger.debug(set_color('map_source_item_field_to_target', 'blue'))
                self._remap_fields(alias, iid_remap_dict)

    def remap_others_id(self):

        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            self._remap(remap_list)

    def _remap_fields(self, field_names, map_dict):
        for field_name in field_names:
            self.field2id_token[field_name] = list(map_dict.keys())
            self.field2token_id[field_name] = map_dict
            if field_name in self.inter_feat.columns:
                self.inter_feat[field_name] = self.inter_feat[field_name].map(map_dict)
            if self.user_feat is not None and field_name in self.user_feat.columns:
                self.user_feat[field_name] = self.user_feat[field_name].map(map_dict)
            if self.item_feat is not None and field_name in self.item_feat.columns:
                self.user_feat[field_name] = self.item_feat[field_name].map(map_dict)

    def data_process_after_remap(self):
        self._user_item_feat_preparation()
        self._fill_nan()
        self._set_label_by_threshold()
        self._normalize()
        self._preload_weight_matrix()
        self._rename_columns()

    def _user_item_feat_preparation(self):
        """Sort :attr:`user_feat` and :attr:`item_feat` by ``user_id`` or ``item_id``.
        Missing values will be filled later.
        """
        if self.user_feat is not None:
            new_user_df = pd.DataFrame({self.uid_field: list(self.field2token_id[self.uid_field].values())})
            self.user_feat = pd.merge(new_user_df, self.user_feat, on=self.uid_field, how='left')
            self.logger.debug(set_color('ordering user features by user id.', 'green'))
        if self.item_feat is not None:
            new_item_df = pd.DataFrame({self.iid_field: list(self.field2token_id[self.iid_field].values())})
            self.item_feat = pd.merge(new_item_df, self.item_feat, on=self.iid_field, how='left')
            self.logger.debug(set_color('ordering item features by item id.', 'green'))


class CrossDomainDataset():
    """:class:`CrossDomainDataset` is based on :class:`~recbole_cdr.data.dataset.dataset.Dataset`,
    and load both `SourceDataset` and `TargetDataset` additionally.

    Users and items in both dataset are remapped together with specially.
    All users (or items) are remapped into three consecutive ID sections.

    - users (or items) that exist both in source dataset and target dataset.
    - users (or items) that only exist in source dataset.
    - users (or items) that only exist in target dataset.


    Attributes:
        source_dataset_name (str): Name of source dataset.

        target_dataset_name (str): Name of target dataset.

        source_dataset_path (str): Local file path of source dataset.

        target_dataset_path (str): Local file path of target dataset.

        field2type (dict): Dict mapping feature name (str) to its type (:class:`~recbole.utils.enum_type.FeatureType`).

        source_uid_field (str or None): The same as ``config['source_dataset']['USER_ID_FIELD']``.

        source_iid_field (str or None): The same as ``config['source_dataset']['ITEM_ID_FIELD']``.

        source_label_field (str or None): The same as ``config['source_dataset']['LABEL_FIELD']``.

        source_time_field (str or None): The same as ``config['source_dataset']['TIME_FIELD']``.

        source_inter_feat (:class:`Interaction`): Internal data structure stores the interactions in source domain.
            It's loaded from file ``.inter`` in the `source_dataset_path`.

        source_user_feat (:class:`Interaction` or None): Internal data structure stores the user features in source domain.
            It's loaded from file ``.user`` in the `source_dataset_path` if existed.

        source_item_feat (:class:`Interaction` or None): Internal data structure stores the item features in source domain.
            It's loaded from file ``.item`` if existed.

        source_feat_name_list (list): A list contains all the features' name in source domain (:class:`str`).

        target_uid_field (str or None): The same as ``config['target_dataset']['USER_ID_FIELD']``.

        target_iid_field (str or None): The same as ``config['target_dataset']['ITEM_ID_FIELD']``.

        target_label_field (str or None): The same as ``config['target_dataset']['LABEL_FIELD']``.

        target_time_field (str or None): The same as ``config['target_dataset']['TIME_FIELD']``.

        target_inter_feat (:class:`Interaction`): Internal data structure stores the interactions in target domain.
            It's loaded from file ``.inter`` in the `target_dataset_path`.

        target_user_feat (:class:`Interaction` or None): Internal data structure stores the user features in target domain.
            It's loaded from file ``.user`` in the `target_dataset_path` if existed.

        target_item_feat (:class:`Interaction` or None): Internal data structure stores the item features in target domain.
            It's loaded from file ``.item`` if existed.

        target_feat_name_list (list): A list contains all the features' name in target domain (:class:`str`).

    """

    def __init__(self, config):
        assert 'source_domain' in config and 'target_domain' in config
        self.config = config
        self.logger = getLogger()
        self.logger.debug(set_color('Source Domain', 'blue'))
        config.update(config['source_domain'])
        self.source_domain_dataset = CrossDomainSingleDataset(config, domain='source')

        self.logger.debug(set_color('Target Domain', 'red'))
        config.update(config['target_domain'])
        self.target_domain_dataset = CrossDomainSingleDataset(config, domain='target')

        self.user_link_dict = None
        self.item_link_dict = None
        self._load_data(config['user_link_file_path'], config['item_link_file_path'])

        # token link remap
        self.source_domain_dataset.remap_user_item_id(self.user_link_dict, self.item_link_dict)

        # user and item ID remap
        self.source_user_ID_remap_dict, self.source_item_ID_remap_dict, \
        self.target_user_ID_remap_dict, self.target_item_ID_remap_dict = self.calculate_user_item_from_both_domain()
        self.source_domain_dataset.remap_user_item_id(self.source_user_ID_remap_dict, self.source_item_ID_remap_dict)
        self.target_domain_dataset.remap_user_item_id(self.target_user_ID_remap_dict, self.target_item_ID_remap_dict)

        # other fields remap
        self.source_domain_dataset.remap_others_id()
        self.target_domain_dataset.remap_others_id()

        # other data process
        self.source_domain_dataset.data_process_after_remap()
        self.target_domain_dataset.data_process_after_remap()

    def calculate_user_item_from_both_domain(self):
        source_user_set = set(self.source_domain_dataset.inter_feat[self.source_domain_dataset.uid_field])
        target_user_set = set(self.target_domain_dataset.inter_feat[self.target_domain_dataset.uid_field])

        if self.source_domain_dataset.user_feat is not None:
            source_user_set = source_user_set | set(
                self.source_domain_dataset.user_feat[self.source_domain_dataset.uid_field])

        if self.target_domain_dataset.user_feat is not None:
            target_user_set = target_user_set | set(
                self.target_domain_dataset.user_feat[self.target_domain_dataset.uid_field])

        overlap_user = source_user_set & target_user_set
        source_only_user = source_user_set - overlap_user
        target_only_user = target_user_set - overlap_user

        self.num_overlap_user = len(overlap_user) + 1
        self.num_source_only_user = len(source_only_user)
        self.num_target_only_user = len(target_only_user)

        self.num_total_user = self.num_overlap_user + self.num_source_only_user + self.num_target_only_user


        overlap_user_remap_dict = dict(zip(overlap_user, range(1, self.num_overlap_user)))
        overlap_user_remap_dict['[PAD]'] = 0
        target_only_user_remap_dict = dict(
            zip(target_only_user,
                range(self.num_overlap_user, self.num_overlap_user + self.num_target_only_user)))
        source_only_user_remap_dict = dict(
            zip(source_only_user, range(self.num_overlap_user + self.num_target_only_user, self.num_total_user)))

        source_user_remap_dict = ChainMap(overlap_user_remap_dict, source_only_user_remap_dict)
        target_user_remap_dict = ChainMap(overlap_user_remap_dict, target_only_user_remap_dict)

        source_item_set = set(self.source_domain_dataset.inter_feat[self.source_domain_dataset.iid_field])
        target_item_set = set(self.target_domain_dataset.inter_feat[self.target_domain_dataset.iid_field])

        if self.source_domain_dataset.item_feat is not None:
            source_item_set = source_item_set | set(
                self.source_domain_dataset.item_feat[self.source_domain_dataset.uid_field])

        if self.target_domain_dataset.item_feat is not None:
            target_item_set = target_item_set | set(
                self.target_domain_dataset.item_feat[self.target_domain_dataset.uid_field])

        overlap_item = source_item_set & target_item_set
        source_only_item = source_item_set - overlap_item
        target_only_item = target_item_set - overlap_item

        self.num_overlap_item = len(overlap_item) + 1
        self.num_source_only_item = len(source_only_item)
        self.num_target_only_item = len(target_only_item)

        self.num_total_item = self.num_overlap_item + self.num_source_only_item + self.num_target_only_item

        overlap_item_remap_dict = dict(zip(overlap_item, range(1, self.num_overlap_item)))
        overlap_item_remap_dict['[PAD]'] = 0
        target_only_item_remap_dict = dict(
            zip(target_only_item,
                range(self.num_overlap_item, self.num_overlap_item + self.num_target_only_item)))
        source_only_item_remap_dict = dict(
            zip(source_only_item, range(self.num_overlap_item + self.num_target_only_item, self.num_total_item)))

        source_item_remap_dict = ChainMap(overlap_item_remap_dict, source_only_item_remap_dict)
        target_item_remap_dict = ChainMap(overlap_item_remap_dict, target_only_item_remap_dict)

        return source_user_remap_dict, source_item_remap_dict, target_user_remap_dict, target_item_remap_dict

    def _load_data(self, user_link_file_path, item_link_file_path):

        if user_link_file_path:
            self.source_user_field = self.config['SOURCE_USER_ID_FIELD']
            self.target_user_field = self.config['TARGET_USER_ID_FIELD']
            self._check_field('source_user_field', 'target_user_field')
            self.user_link_dict = self._load_link(user_link_file_path, between='user')

        if item_link_file_path:
            self.source_item_field = self.config['SOURCE_ITEM_ID_FIELD']
            self.target_item_field = self.config['TARGET_ITEM_ID_FIELD']
            self._check_field('source_item_field', 'target_item_field')
            self.item_link_dict = self._load_link(item_link_file_path, between='item')

    def __str__(self):
        info = [
            f'Source domain: {self.source_domain_dataset.__str__()}',
            f'Target domain: {self.target_domain_dataset.__str__()}',
            f'Num of overlapped user: {self.num_overlap_user}',
            f'Num of overlapped item: {self.num_overlap_item}',
        ]  # yapf: disable
        return '\n'.join(info)

    def _load_link(self, link_file_path, between='user'):
        self.logger.debug(set_color(f'Loading ID link between cross domain.', 'green'))
        if not os.path.isfile(link_file_path):
            raise ValueError(f'link file not found. Please check the path:[{link_file_path}].')
        link_df = self._load_link_file(link_file_path, between + '_link')
        self._check_link(link_df, between)

        source2target = {}
        if between == 'user':
            source_field = self.source_user_field
            target_field = self.target_user_field
        else:
            source_field = self.source_item_field
            target_field = self.target_item_field
        for source_id, target_id in zip(link_df[source_field].values, link_df[target_field].values):
            source2target[source_id] = target_id
        return source2target

    def _check_link(self, link, between='user'):
        if between == 'user':
            link_warn_message = 'link data between users requires field [{}]'
            assert self.source_user_field in link, link_warn_message.format(self.source_user_field)
            assert self.target_user_field in link, link_warn_message.format(self.target_user_field)
        else:
            link_warn_message = 'link data between item requires field [{}]'
            assert self.source_item_field in link, link_warn_message.format(self.source_item_field)
            assert self.target_item_field in link, link_warn_message.format(self.target_item_field)

    def _load_link_file(self, filepath, source):
        """Load links according to source into :class:`pandas.DataFrame`.
        Args:
            filepath (str): path of input file.
            source (FeatureSource or str): source of input file.

        Returns:
            pandas.DataFrame: Loaded link

        """
        self.logger.debug(set_color(f'Loading link from [{filepath}] (source: [{source}]).', 'green'))

        field_separator = self.config['field_separator']
        columns = []
        usecols = []
        dtype = {}
        encoding = self.config['encoding']
        with open(filepath, 'r', encoding=encoding) as f:
            head = f.readline()[:-1]
        for field_type in head.split(field_separator):
            field, ftype = field_type.split(':')
            try:
                ftype = FeatureType(ftype)
            except ValueError:
                raise ValueError(f'Type {ftype} from field {field} is not supported.')

            if not ftype == FeatureType.TOKEN:
                continue
            columns.append(field)
            usecols.append(field_type)
            dtype[field_type] = np.float64 if ftype == FeatureType.FLOAT else str

        if len(columns) == 0:
            self.logger.warning(f'No columns has been loaded from [{source}]')
            return None

        df = pd.read_csv(
            filepath, delimiter=field_separator, usecols=usecols, dtype=dtype, encoding=encoding, engine='python'
        )
        df.columns = columns
        return df

    def build(self):
        """Processing dataset in target domain according to evaluation setting, including Group, Order and Split.
        See :class:`~recbole_cdr.config.eval_setting.EvalSetting` for details.

        Returns:
            list: List of built :class:`Dataset`.
        """
        target_domain_train_dataset, target_domain_valid_dataset, target_domain_test_dataset \
            = self.target_domain_dataset.build()

        source_domain_train_dataset = copy.copy(self.source_domain_dataset)
        source_domain_train_dataset._change_feat_format()

        return [source_domain_train_dataset, target_domain_train_dataset,
                target_domain_valid_dataset, target_domain_test_dataset]

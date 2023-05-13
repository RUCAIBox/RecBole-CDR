# @Time   : 2022/3/8
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn
# UPDATE
# @Time   : 2022/4/9
# @Author : Gaowei Zhang
# @email  : 1462034631@qq.com

"""
recbole_cdr.data.dataset
##########################
"""

import os
from collections import ChainMap
import torch

import numpy as np
import pandas as pd
from logging import getLogger
from scipy.sparse import coo_matrix

from recbole.data.dataset import Dataset
from recbole.utils import FeatureSource, FeatureType, set_color
from recbole_cdr.utils import get_keys_from_chainmap_by_order


class CrossDomainSingleDataset(Dataset):
    def __init__(self, config, domain='source'):
        self.domain = domain
        super().__init__(config)

    def _data_processing(self):
        """Data preprocessing, including:

        - Data filtering
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
        """Remap the ids of users or items in the two dataset.

        Args:
            uid_remap_dict (dict): The dict whose keys are the users' id in source domain
                                    and values are users' id in target domain.
            iid_remap_dict (dict): The dict whose keys are the items' id in source domain
                                    and values are items' id in target domain.
        """

        for alias in self.alias.values():
            if uid_remap_dict and self.uid_field in alias:
                self.logger.debug(set_color('map_source_user_field_to_target', 'blue'))
                self._remap_fields(alias, uid_remap_dict)
            if iid_remap_dict and self.iid_field in alias:
                self.logger.debug(set_color('map_source_item_field_to_target', 'blue'))
                self._remap_fields(alias, iid_remap_dict)

    def remap_others_id(self):
        """Remap the other data fields that share the ids with users or items.
        """
        for field in self._rest_fields:
            remap_list = self._get_remap_list(np.array([field]))
            self._remap(remap_list)

    def _remap_fields(self, field_names, map_dict):
        """Remap the ids in targeted fields
        Args:
            field_names (list of str): The list of field names.
            map_dict (dict): The dict whose keys are the original ids and values are the new ids.
        """
        for field_name in field_names:
            self.field2id_token[field_name] = get_keys_from_chainmap_by_order(map_dict)
            self.field2token_id[field_name] = map_dict
            if field_name in self.inter_feat.columns:
                self.inter_feat[field_name] = self.inter_feat[field_name].map(lambda x: map_dict.get(x, x))
            if self.user_feat is not None and field_name in self.user_feat.columns:
                self.user_feat[field_name] = self.user_feat[field_name].map(lambda x: map_dict.get(x, x))
            if self.item_feat is not None and field_name in self.item_feat.columns:
                self.item_feat[field_name] = self.item_feat[field_name].map(lambda x: map_dict.get(x, x))

    def data_process_after_remap(self):
        """Data preprocessing, including:
            - Missing value imputation
            - Normalization
            - Preloading weights initialization
        """
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

    def get_sparse_matrix(self, user_num, item_num, form='coo', value_field=None):
        """Get sparse matrix that describe relations between two fields.

        Source and target should be token-like fields.

        Sparse matrix has shape (``self.num(source_field)``, ``self.num(target_field)``).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = df_feat[value_field][src, tgt]``.

        Args:
            user_num (int): Number of users.
            item_num (int): Number of items.
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        src = self.inter_feat[self.uid_field]
        tgt = self.inter_feat[self.iid_field]
        if value_field is None:
            data = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.inter_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `df_feat`\'s features.')
            data = self.inter_feat[value_field]
        mat = coo_matrix((data, (src, tgt)), shape=(user_num, item_num))

        if form == 'coo':
            return mat
        elif form == 'csr':
            return mat.tocsr()
        else:
            raise NotImplementedError(f'Sparse matrix format [{form}] has not been implemented.')

    def get_history_matrix(self, user_num, item_num, row, value_field=None):
        """Get dense matrix describe user/item's history interaction records.

        ``history_matrix[i]`` represents ``i``'s history interacted ids.

        ``history_value[i]`` represents ``i``'s history interaction records' values.
            ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            user_num (int): Number of users.
            item_num (int): Number of items.
            row (str): ``user`` or ``item``.
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        self._check_field('uid_field', 'iid_field')

        user_ids, item_ids = self.inter_feat[self.uid_field].numpy(), self.inter_feat[self.iid_field].numpy()
        if value_field is None:
            values = np.ones(len(self.inter_feat))
        else:
            if value_field not in self.inter_feat:
                raise ValueError(f'Value_field [{value_field}] should be one of `inter_feat`\'s features.')
            values = self.inter_feat[value_field].numpy()

        if row == 'user':
            row_num, max_col_num = user_num, item_num
            row_ids, col_ids = user_ids, item_ids
        else:
            row_num, max_col_num = item_num, user_num
            row_ids, col_ids = item_ids, user_ids

        history_len = np.zeros(row_num, dtype=np.int64)
        for row_id in row_ids:
            history_len[row_id] += 1

        col_num = np.max(history_len)
        if col_num > max_col_num * 0.2:
            self.logger.warning(
                f'Max value of {row}\'s history interaction records has reached '
                f'{col_num / max_col_num * 100}% of the total.'
            )

        history_matrix = np.zeros((row_num, col_num), dtype=np.int64)
        history_value = np.zeros((row_num, col_num))
        history_len[:] = 0
        for row_id, value, col_id in zip(row_ids, values, col_ids):
            history_matrix[row_id, history_len[row_id]] = col_id
            history_value[row_id, history_len[row_id]] = value
            history_len[row_id] += 1

        return torch.LongTensor(history_matrix), torch.FloatTensor(history_value), torch.LongTensor(history_len)

    def split_train_valid(self):
        self._change_feat_format()

        if self.benchmark_filename_list is not None:
            cumsum = list(np.cumsum(self.file_size_list))
            datasets = [self.copy(self.inter_feat[start:end]) for start, end in zip([0] + cumsum[:-1], cumsum)]
            return datasets

        # ordering
        ordering_args = self.config['eval_args']['order']
        if ordering_args == 'RO':
            self.shuffle()
        elif ordering_args == 'TO':
            self.sort(by=self.time_field)
        else:
            raise NotImplementedError(f'The ordering_method [{ordering_args}] has not been implemented.')

        # splitting & grouping
        split_args = self.config['eval_args']['split_valid']
        if split_args is None:
            raise ValueError('The split_args in eval_args should not be None.')
        if not isinstance(split_args, dict):
            raise ValueError(f'The split_args [{split_args}] should be a dict.')

        split_mode = list(split_args.keys())[0]
        assert len(split_args.keys()) == 1
        group_by = self.config['eval_args']['group_by']
        if split_mode == 'RS':
            if not isinstance(split_args['RS'], list):
                raise ValueError(f'The value of "RS" [{split_args}] should be a list.')
            if group_by is None or group_by.lower() == 'none':
                datasets = self.split_by_ratio(split_args['RS'], group_by=None)
            elif group_by == 'user':
                datasets = self.split_by_ratio(split_args['RS'], group_by=self.uid_field)
            else:
                raise NotImplementedError(f'The grouping method [{group_by}] has not been implemented.')
        else:
            raise NotImplementedError(f'The splitting_method [{split_mode}] has not been implemented.')

        return datasets


class CrossDomainDataset:
    """:class:`CrossDomainDataset` is based on :class:`~recbole.data.dataset.dataset.Dataset`,
    and load both `SourceDataset` and `TargetDataset` additionally.

    Users and items in both dataset are remapped together.
    All users (or items) are remapped into three consecutive ID sections.

    - users (or items) that exist both in source dataset and target dataset.
    - users (or items) that only exist in source dataset.
    - users (or items) that only exist in target dataset.
    """

    def __init__(self, config):
        assert 'source_domain' in config and 'target_domain' in config
        self.config = config
        self.logger = getLogger()
        self.train_modes = config['train_modes']
        self.logger.debug(set_color('Source Domain', 'blue'))
        source_config = config.update(config['source_domain'])
        self.source_domain_dataset = CrossDomainSingleDataset(source_config, domain='source')

        self.logger.debug(set_color('Target Domain', 'red'))
        target_config = config.update(config['target_domain'])
        self.target_domain_dataset = CrossDomainSingleDataset(target_config, domain='target')

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
        if self.num_overlap_user > 1:
            self.overlap_dataset = CrossDomainOverlapDataset(config, self.num_overlap_user)
        else:
            self.overlap_dataset = CrossDomainOverlapDataset(config, self.num_overlap_item)
        self.overlap_id_field = self.overlap_dataset.overlap_id_field

    def calculate_user_item_from_both_domain(self):
        """Prepare the remap dict for the users and items in both domain.

        Returns:
            source_user_remap_dict(dict): the dict for source domain whose keys are user original ids
                                            and values are mapped ids.
            source_item_remap_dict(dict): the dict for source domain whose keys are item original ids
                                            and values are mapped ids.
            target_user_remap_dict(dict): the dict for target domain whose keys are user original ids
                                            and values are mapped ids.
            target_item_remap_dict(dict): the dict for target domain whose keys are item original ids
                                            and values are mapped ids.

        """
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

        overlap_user = list(overlap_user)
        source_only_user = list(source_only_user)
        target_only_user = list(target_only_user)
        while np.nan in source_only_user:
            source_only_user.remove(np.nan)
        while np.nan in target_only_user:
            target_only_user.remove(np.nan)
        overlap_user.sort()
        source_only_user.sort()
        target_only_user.sort()

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

        overlap_item = list(overlap_item)
        source_only_item = list(source_only_item)
        target_only_item = list(target_only_item)
        while np.nan in source_only_item:
            source_only_item.remove(np.nan)
        while np.nan in target_only_item:
            target_only_item.remove(np.nan)

        overlap_item.sort()
        source_only_item.sort()
        target_only_item.sort()

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
            self.source_user_field = self.source_domain_dataset.uid_field
            self.target_user_field = self.target_domain_dataset.uid_field
            self.user_link_dict = self._load_link(user_link_file_path, between='user')

        if item_link_file_path:
            self.source_item_field = self.source_domain_dataset.iid_field
            self.target_item_field = self.target_domain_dataset.iid_field
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
        """ Check whether the link file is in the correct format.

        Args:
            link (str): path of input file.
            between (str): user of item that to be linked. default to 'user'

        """
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

        self.overlap_dataset._change_feat_format()

        source_split_flag = self.config['source_split']

        if not source_split_flag:
            source_domain_train_dataset = self.source_domain_dataset
            source_domain_train_dataset._change_feat_format()

            return [source_domain_train_dataset, None, target_domain_train_dataset,
                    target_domain_valid_dataset, target_domain_test_dataset]
        else:
            source_domain_train_dataset, source_domain_valid_dataset = self.source_domain_dataset.split_train_valid()
            return [source_domain_train_dataset, source_domain_valid_dataset, target_domain_train_dataset,
                    target_domain_valid_dataset, target_domain_test_dataset]

    def inter_matrix(self, form='coo', value_field=None, domain='source'):
        """Get sparse matrix that describe interactions between user_id and item_id.

        Sparse matrix has shape (user_num, item_num).

        For a row of <src, tgt>, ``matrix[src, tgt] = 1`` if ``value_field`` is ``None``,
        else ``matrix[src, tgt] = self.inter_feat[src, tgt]``.

        Args:
            form (str, optional): Sparse matrix format. Defaults to ``coo``.
            value_field (str, optional): Data of sparse matrix, which should exist in ``df_feat``.
                Defaults to ``None``.
            domain (str, optional): Identifier string of the domain. Defaults to ``source``

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        if domain == 'source':
            if not self.source_domain_dataset.uid_field or not self.source_domain_dataset.iid_field:
                raise ValueError('source dataset does not exist uid/iid, thus can not converted to sparse matrix.')
            return self.source_domain_dataset.get_sparse_matrix(self.num_total_user, self.num_total_item, form, value_field)
        else:
            if not self.target_domain_dataset.uid_field or not self.target_domain_dataset.iid_field:
                raise ValueError('target dataset does not exist uid/iid, thus can not converted to sparse matrix.')
            return self.target_domain_dataset.get_sparse_matrix(self.num_total_user, self.num_total_item, form, value_field)

    def history_user_matrix(self, value_field=None, domain='source'):
        """Get dense matrix describe item's history interaction records.

        ``history_matrix[i]`` represents item ``i``'s history interacted user_id.

        ``history_value[i]`` represents item ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of item ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.
            domain (str, optional): Identifier string of the domain. Defaults to ``source``

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        if domain == 'source':
            return self.source_domain_dataset.get_history_matrix(self.num_total_user, self.num_total_item,
                                                                 row='item', value_field=value_field)
        else:
            return self.target_domain_dataset.get_history_matrix(self.num_total_user, self.num_total_item,
                                                                 row='item', value_field=value_field)

    def history_item_matrix(self, value_field=None, domain='source'):
        """Get dense matrix describe item's history interaction records.

        ``history_matrix[i]`` represents user ``i``'s history interacted item_id.

        ``history_value[i]`` represents user ``i``'s history interaction records' values,
        ``0`` if ``value_field = None``.

        ``history_len[i]`` represents number of item ``i``'s history interaction records.

        ``0`` is used as padding.

        Args:
            value_field (str, optional): Data of matrix, which should exist in ``self.inter_feat``.
                Defaults to ``None``.
            domain (str, optional): Identifier string of the domain. Defaults to ``source``

        Returns:
            tuple:
                - History matrix (torch.Tensor): ``history_matrix`` described above.
                - History values matrix (torch.Tensor): ``history_value`` described above.
                - History length matrix (torch.Tensor): ``history_len`` described above.
        """
        if domain == 'source':
            return self.source_domain_dataset.get_history_matrix(self.num_total_user, self.num_total_item,
                                                                 row='user', value_field=value_field)
        else:
            return self.target_domain_dataset.get_history_matrix(self.num_total_user, self.num_total_item,
                                                                 row='user', value_field=value_field)


class CrossDomainOverlapDataset(Dataset):
    """:class:`CrossDomainOverlapDataset` contains the data of overlapped users or items.
    """

    def __init__(self, config, num_overlap):
        self.num_overlap = num_overlap
        super(CrossDomainOverlapDataset, self).__init__(config)

    def _from_scratch(self):
        self.logger.debug(set_color(f'Loading {self.__class__} from scratch.', 'green'))

        self._get_preset()
        self._load_data(self.dataset_name, self.dataset_path)
        self._data_processing()

    def _build_feat_name_list(self):
        feat_name_list = ['overlap_feat']
        return feat_name_list

    def _data_processing(self):
        self.feat_name_list = self._build_feat_name_list()

    def __len__(self):
        return self.num_overlap

    def shuffle(self):
        self.overlap_feat.shuffle()

    def _load_data(self, token, dataset_path):
        """Rewrite the function. data is constructed not loaded from files.

        """
        field = 'overlap'
        ftype = FeatureType.TOKEN
        self.overlap_id_field = field
        self.field2type[field] = ftype
        self.overlap_feat = {}
        overlap_data = np.arange(self.num_overlap)
        np.random.shuffle(overlap_data)
        self.overlap_feat[field] = pd.DataFrame(np.array(overlap_data))

    def __getitem__(self, index, join=True):
        df = self.overlap_feat[index]
        return df

    def __str__(self):
        info = [set_color(self.dataset_name, 'pink'),
                set_color('The number of overlap idx', 'blue') + f': {self.num_overlap}',
                set_color('Remain Fields', 'blue') + f': {list(self.field2type)}']
        return '\n'.join(info)

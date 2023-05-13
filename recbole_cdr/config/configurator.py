# @Time   : 2022/3/12
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn


"""
recbole_cdr.config.cd_configurator
################################
"""
import os
import copy
from recbole.config.configurator import Config
from recbole.evaluator import metric_types, smaller_metrics
from recbole.utils import EvaluatorType, ModelType, InputType

from recbole_cdr.utils import get_model, train_mode2state


class CDRConfig(Config):
    """ Configurator module that load the defined parameters.

    Configurator module will first load the default parameters from the fixed properties in RecBole and then
    load parameters from the external input.

    External input supports three kind of forms: config file, command line and parameter dictionaries.

    - config file: It's a file that record the parameters to be modified or added. It should be in ``yaml`` format,
      e.g. a config file is 'example.yaml', the content is:

        learning_rate: 0.001

        train_batch_size: 2048

    - command line: It should be in the format as '---learning_rate=0.001'

    - parameter dictionaries: It should be a dict, where the key is parameter name and the value is parameter value,
      e.g. config_dict = {'learning_rate': 0.001}

    Configuration module allows the above three kind of external input format to be used together,
    the priority order is as following:

    command line > parameter dictionaries > config file

    e.g. If we set learning_rate=0.01 in config file, learning_rate=0.02 in command line,
    learning_rate=0.03 in parameter dictionaries.

    Finally the learning_rate is equal to 0.02.
    """

    def __init__(self, model=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str/CrossDomainRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """
        self.compatibility_settings()
        self._init_parameters_category()
        self.parameters['Dataset'] += ['source_domain', 'target_domain']
        self.yaml_loader = self._build_yaml_loader()
        self.file_config_dict = self._remove_domain_prefix(
            self._load_config_files(config_file_list))
        self.variable_config_dict = self._remove_domain_prefix(
            self._load_variable_config_dict(config_dict))
        self.cmd_config_dict = self._remove_domain_prefix(self._load_cmd_line())
        self._merge_external_config_dict()

        self.model, self.model_class, self.dataset = self._get_model_and_dataset(model)
        self._load_internal_config_dict(self.model, self.model_class, self.dataset)
        self.final_config_dict = self._get_final_config_dict()
        self._set_default_parameters()
        self._init_device()
        self._set_train_neg_sample_args()
        self._set_eval_neg_sample_args()
        self.dataset = self._check_cross_domain()

    def _check_cross_domain(self):
        """r Check the parameters whether in the format of Cross-domain Recommendation and return the formatted dataset

        """
        assert 'source_domain' in self.final_config_dict or 'target_domain' in self.final_config_dict
        try:
            source_dataset_name = self.final_config_dict['source_domain']['dataset']
            target_dataset_name = self.final_config_dict['target_domain']['dataset']
        except KeyError:
            raise KeyError(
                'If you want to run cross-domain recommender, '
                'name of both source domain and target domain should be specified in config file.'
            )
        if source_dataset_name == 'ml-100k' or source_dataset_name == 'ml-1m':
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.final_config_dict['source_domain']['data_path'] = os.path.join(current_path,
                                                                                '../dataset_example/' + source_dataset_name)
        else:
            if 'data_path' not in self.final_config_dict['source_domain']:
                data_path = self.final_config_dict['data_path']
            else:
                data_path = self.final_config_dict['source_domain']['data_path']
            self.final_config_dict['source_domain']['data_path'] = os.path.join(data_path, source_dataset_name)

        if target_dataset_name == 'ml-100k' or target_dataset_name == 'ml-1m':
            current_path = os.path.dirname(os.path.realpath(__file__))
            self.final_config_dict['target_domain']['data_path'] = os.path.join(current_path,
                                                                                '../dataset_example/' + target_dataset_name)
        else:
            if 'data_path' not in self.final_config_dict['target_domain']:
                data_path = self.final_config_dict['data_path']
            else:
                data_path = self.final_config_dict['target_domain']['data_path']
            self.final_config_dict['target_domain']['data_path'] = os.path.join(data_path, target_dataset_name)

        self.final_config_dict['dataset'] = {'source_domain': source_dataset_name,
                                             'target_domain': target_dataset_name}
        return self.final_config_dict['dataset']

    def _get_model_and_dataset(self, model, dataset=None):

        if model is None:
            try:
                model = self.external_config_dict['model']
            except KeyError:
                raise KeyError(
                    'model need to be specified in at least one of the these ways: '
                    '[model variable, config file, config dict, command line] '
                )
        if not isinstance(model, str):
            final_model_class = model
            final_model = model.__name__
        else:
            final_model = model
            final_model_class = get_model(final_model)

        return final_model, final_model_class, None

    def _load_internal_config_dict(self, model, model_class, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, '../properties/overall.yaml')
        model_init_file = os.path.join(current_path, '../properties/model/' + model + '.yaml')
        sample_init_file = os.path.join(current_path, '../properties/dataset/sample.yaml')

        self.internal_config_dict = dict()
        for file in [overall_init_file, model_init_file, sample_init_file]:
            if os.path.isfile(file):
                self._update_internal_config_dict(file)

        self.internal_config_dict['MODEL_TYPE'] = model_class.type

    def _set_default_parameters(self):
        self.final_config_dict['model'] = self.model

        if hasattr(self.model_class, 'input_type'):
            self.final_config_dict['MODEL_INPUT_TYPE'] = self.model_class.input_type
        elif 'loss_type' in self.final_config_dict:
            if self.final_config_dict['loss_type'] in ['CE']:
                if self.final_config_dict['MODEL_TYPE'] == ModelType.SEQUENTIAL and \
                   self.final_config_dict['neg_sampling'] is not None:
                    raise ValueError(f"neg_sampling [{self.final_config_dict['neg_sampling']}] should be None "
                                     f"when the loss_type is CE.")
                self.final_config_dict['MODEL_INPUT_TYPE'] = InputType.POINTWISE
            elif self.final_config_dict['loss_type'] in ['BPR']:
                self.final_config_dict['MODEL_INPUT_TYPE'] = InputType.PAIRWISE
        else:
            raise ValueError('Either Model has attr \'input_type\',' 'or arg \'loss_type\' should exist in config.')

        metrics = self.final_config_dict['metrics']
        if isinstance(metrics, str):
            self.final_config_dict['metrics'] = [metrics]

        eval_type = set()
        for metric in self.final_config_dict['metrics']:
            if metric.lower() in metric_types:
                eval_type.add(metric_types[metric.lower()])
            else:
                raise NotImplementedError(f"There is no metric named '{metric}'")
        if len(eval_type) > 1:
            raise RuntimeError('Ranking metrics and value metrics can not be used at the same time.')
        self.final_config_dict['eval_type'] = eval_type.pop()

        if self.final_config_dict['MODEL_TYPE'] == ModelType.SEQUENTIAL and not self.final_config_dict['repeatable']:
            raise ValueError('Sequential models currently only support repeatable recommendation, '
                             'please set `repeatable` as `True`.')

        valid_metric = self.final_config_dict['valid_metric'].split('@')[0]
        self.final_config_dict['valid_metric_bigger'] = False if valid_metric.lower() in smaller_metrics else True

        topk = self.final_config_dict['topk']
        if isinstance(topk, (int, list)):
            if isinstance(topk, int):
                topk = [topk]
            for k in topk:
                if k <= 0:
                    raise ValueError(
                        f'topk must be a positive integer or a list of positive integers, but get `{k}`'
                    )
            self.final_config_dict['topk'] = topk
        else:
            raise TypeError(f'The topk [{topk}] must be a integer, list')

        if 'additional_feat_suffix' in self.final_config_dict:
            ad_suf = self.final_config_dict['additional_feat_suffix']
            if isinstance(ad_suf, str):
                self.final_config_dict['additional_feat_suffix'] = [ad_suf]

        # eval_args checking
        default_eval_args = {
            'split': {'RS': [0.8, 0.1, 0.1]},
            'order': 'RO',
            'group_by': 'user',
            'mode': 'full'
        }
        if not isinstance(self.final_config_dict['eval_args'], dict):
            raise ValueError(f"eval_args:[{self.final_config_dict['eval_args']}] should be a dict.")
        for op_args in default_eval_args:
            if op_args not in self.final_config_dict['eval_args']:
                self.final_config_dict['eval_args'][op_args] = default_eval_args[op_args]

        if (self.final_config_dict['eval_args']['mode'] == 'full'
                and self.final_config_dict['eval_type'] == EvaluatorType.VALUE):
            raise NotImplementedError('Full sort evaluation do not match value-based metrics!')

        # training_mode args
        train_scheme = []
        train_epochs = []
        for train_arg in self.final_config_dict['train_epochs']:
            scheme, epoch = train_arg.split(':')
            if scheme not in train_mode2state:
                raise ValueError(f"[{scheme}] is not a supported training mode.")
            train_scheme.append(scheme)
            train_epochs.append(epoch)
        self.final_config_dict['train_modes'] = train_scheme
        self.final_config_dict['epoch_num'] = train_epochs
        source_split_flag = True if 'SOURCE' in train_scheme else False
        self.final_config_dict['source_split'] = source_split_flag
        self.final_config_dict['epochs'] = int(train_epochs[0])

    @staticmethod
    def _remove_domain_prefix(config_dict):
        if 'source_domain' not in config_dict:
            config_dict['source_domain'] = dict()
        if 'target_domain' not in config_dict:
            config_dict['target_domain'] = dict()
        for key in list(config_dict.keys()):
            if key.startswith('source_') and not key.startswith('source_domain'):
                config_dict['source_domain'][key[7:]] = copy.copy(config_dict[key])
                config_dict.pop(key)
            elif key.startswith('target_') and not key.startswith('target_domain'):
                config_dict['target_domain'][key[7:]] = copy.copy(config_dict[key])
                config_dict.pop(key)
        return config_dict

    def _merge_external_config_dict(self):
        external_config_dict = dict()
        external_source_config_dict, external_target_config_dict = dict(), dict()
        external_config_dict.update(self.file_config_dict)
        external_config_dict.update(self.variable_config_dict)
        external_config_dict.update(self.cmd_config_dict)
        external_source_config_dict.update(self.file_config_dict['source_domain'])
        external_source_config_dict.update(self.variable_config_dict['source_domain'])
        external_source_config_dict.update(self.cmd_config_dict['source_domain'])
        external_target_config_dict.update(self.file_config_dict['target_domain'])
        external_target_config_dict.update(self.variable_config_dict['target_domain'])
        external_target_config_dict.update(self.cmd_config_dict['target_domain'])
        external_config_dict['source_domain'] = external_source_config_dict
        external_config_dict['target_domain'] = external_target_config_dict
        self.external_config_dict = external_config_dict

    def _get_final_config_dict(self):
        final_config_dict = dict()
        final_source_config_dict, final_target_config_dict = dict(), dict()
        final_config_dict.update(self.internal_config_dict)
        final_config_dict.update(self.external_config_dict)
        final_source_config_dict.update(self.internal_config_dict['source_domain'])
        final_source_config_dict.update(self.external_config_dict['source_domain'])
        final_target_config_dict.update(self.internal_config_dict['target_domain'])
        final_target_config_dict.update(self.external_config_dict['target_domain'])
        final_config_dict['source_domain'] = final_source_config_dict
        final_config_dict['target_domain'] = final_target_config_dict
        return final_config_dict

    def update(self, other_config):
        new_config_obj = copy.deepcopy(self)
        for key in other_config:
            new_config_obj.final_config_dict[key] = other_config[key]
        return new_config_obj
    
    def compatibility_settings(self):
        import numpy as np
        np.bool = np.bool_
        np.int = np.int_
        np.float = np.float_
        np.complex = np.complex_
        np.object = np.object_
        np.str = np.str_
        np.long = np.int_
        np.unicode = np.unicode_

# @Time   : 2022/3/12
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn


"""
recbole_cross_domain.config.cd_configurator
################################
"""
import os
from recbole.config.configurator import Config

from recbole_cross_domain.utils import get_model


class CDConfig(Config):
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

    def __init__(self, model=None, dataset=None, config_file_list=None, config_dict=None):
        """
        Args:
            model (str/AbstractRecommender): the model name or the model class, default is None, if it is None, config
            will search the parameter 'model' from the external input as the model name or model class.
            dataset (str): the dataset name, default is None, if it is None, config will search the parameter 'dataset'
            from the external input as the dataset name.
            config_file_list (list of str): the external config file, it allows multiple config files, default is None.
            config_dict (dict): the external parameter dictionaries, default is None.
        """

        super().__init__(model, dataset, config_file_list, config_dict)
        self.dataset = self._check_cross_domain()

    def _check_cross_domain(self):
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

    def _get_model_and_dataset(self, model, dataset):

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

        if dataset is None:
            try:
                final_dataset = self.external_config_dict['dataset']
            except KeyError:
                raise KeyError(
                    'dataset need to be specified in at least one of the these ways: '
                    '[dataset variable, config file, config dict, command line] '
                )
        else:
            final_dataset = dataset

        return final_model, final_model_class, final_dataset

    def _load_internal_config_dict(self, model, model_class, dataset):
        current_path = os.path.dirname(os.path.realpath(__file__))
        overall_init_file = os.path.join(current_path, '../properties/overall.yaml')
        model_init_file = os.path.join(current_path, '../properties/model/' + model + '.yaml')
        sample_init_file = os.path.join(current_path, '../properties/dataset/sample.yaml')
        dataset_init_file = os.path.join(current_path, '../properties/dataset/' + dataset + '.yaml')

        self.internal_config_dict = dict()
        for file in [overall_init_file, model_init_file, sample_init_file, dataset_init_file]:
            if os.path.isfile(file):
                config_dict = self._update_internal_config_dict(file)
                if file == dataset_init_file:
                    self.parameters['Dataset'] += [
                        key for key in config_dict.keys() if key not in self.parameters['Dataset']
                    ]

        self.internal_config_dict['MODEL_TYPE'] = model_class.type


    def update(self, other_config):
        for key in other_config:
            self.final_config_dict[key] = other_config[key]

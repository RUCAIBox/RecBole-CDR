# @Time   : 2022/3/12
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole_cdr.model.crossdomain_recommender
##################################
"""

from recbole.model.abstract_recommender import AbstractRecommender
from recbole_cdr.utils import ModelType


class CrossDomainRecommender(AbstractRecommender):
    """This is a abstract cross-domain recommender. All the cross-domain model should implement this class.
    The base cross-domain recommender class provide the basic dataset and parameters information.
    """

    type = ModelType.CROSSDOMAIN

    def __init__(self, config, dataset):
        super(CrossDomainRecommender, self).__init__()

        # load source dataset info
        self.SOURCE_USER_ID = dataset.source_domain_dataset.uid_field
        self.SOURCE_ITEM_ID = dataset.source_domain_dataset.iid_field
        self.SOURCE_NEG_ITEM_ID = config['source_domain']['NEG_PREFIX'] + self.SOURCE_ITEM_ID
        self.source_num_users = dataset.source_domain_dataset.num(self.SOURCE_USER_ID)
        self.source_num_items = dataset.source_domain_dataset.num(self.SOURCE_ITEM_ID)

        # load target dataset info
        self.TARGET_USER_ID = dataset.target_domain_dataset.uid_field
        self.TARGET_ITEM_ID = dataset.target_domain_dataset.iid_field
        self.TARGET_NEG_ITEM_ID = config['target_domain']['NEG_PREFIX'] + self.TARGET_ITEM_ID
        self.target_num_users = dataset.target_domain_dataset.num(self.TARGET_USER_ID)
        self.target_num_items = dataset.target_domain_dataset.num(self.TARGET_ITEM_ID)

        # load both dataset info
        self.total_num_users = dataset.num_total_user
        self.total_num_items = dataset.num_total_item

        self.overlapped_num_users = dataset.num_overlap_user
        self.overlapped_num_items = dataset.num_overlap_item

        self.OVERLAP_ID = dataset.overlap_id_field

        # load parameters info
        self.device = config['device']

    def set_phase(self, phase):
        pass

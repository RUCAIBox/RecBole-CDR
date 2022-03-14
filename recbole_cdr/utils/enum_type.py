# -*- coding: utf-8 -*-
# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

"""
recbole_cdr.utils.enum_type
#######################
"""

from enum import Enum


class ModelType(Enum):
    """Type of models.

    - ``CROSSDOMAIN``: Cross Domain Recommendation
    """

    CROSSDOMAIN = 1


class CrossDomainDataLoaderState(Enum):
    """States for Cross-domain DataLoader.

    - ``BOTH``: Return both data in source domain and target domain.
    - ``SOURCE``: Only return the data in source domain.
    - ``TARGET``: Only return the data in targe domain.
    """

    BOTH = 1
    SOURCE = 2
    TARGET = 3


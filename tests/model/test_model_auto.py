# -*- coding: utf-8 -*-
# @Time   : 2020/10/24
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

# UPDATE
# @Time    :   2020/11/17
# @Author  :   Xingyu Pan
# @email   :   panxy@ruc.edu.cn

import os
import unittest

from recbole.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


def quick_test(config_dict):
    objective_function(config_dict=config_dict, config_file_list=config_file_list, saved=False)


class TestCrossDomainRecommender(unittest.TestCase):
    def test_cmfrec(self):
        config_dict = {
            'model': 'CMF',
            'config_files': './recbole/properties/ml-1m2ml-100k.yaml',
        }
        quick_test(config_dict)


if __name__ == '__main__':
    unittest.main()

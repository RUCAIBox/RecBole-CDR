import os
import unittest

from recbole_cdr.quick_start import objective_function

current_path = os.path.dirname(os.path.realpath(__file__))
config_file_list = [os.path.join(current_path, 'test_model.yaml')]


def quick_test(config_dict):
    objective_function(config_dict=config_dict, config_file_list=config_file_list, saved=False)


class TestRecommender(unittest.TestCase):
    def test_cmf(self):
        config_dict = {
            'model': 'CMF',
            'train_epochs': ["BOTH:1"],
        }
        quick_test(config_dict)

    def test_clfm(self):
        config_dict = {
            'model': 'CLFM',
            'train_epochs': ["BOTH:1"],
        }
        quick_test(config_dict)

    def test_bitgcf(self):
        config_dict = {
            'model': 'BiTGCF',
            'train_epochs': ["BOTH:1"]
        }
        quick_test(config_dict)

    def test_conet(self):
        config_dict = {
            'model': 'CoNet',
            'train_epochs': ["BOTH:1"],
        }
        quick_test(config_dict)

    def test_deepapf(self):
        config_dict = {
            'model': 'DeepAPF',
            'train_epochs': ["BOTH:1"],
        }
        quick_test(config_dict)

    def test_dtcdr(self):
        config_dict = {
            'model': 'DTCDR',
            'train_epochs': ["BOTH:1"],
        }
        quick_test(config_dict)

    def test_emcdr(self):
        config_dict = {
            'model': 'EMCDR',
            'train_epochs': ["SOURCE:1", "TARGET:1", "BOTH:1"],
        }
        quick_test(config_dict)

    def test_sscdr(self):
        config_dict = {
            'model': 'SSCDR',
            'train_epochs': ["SOURCE:1", "TARGET:1", "BOTH:1"],
        }
        quick_test(config_dict)


    def test_dcdcsr(self):
        config_dict = {
            'model': 'DCDCSR',
            'train_epochs': ["SOURCE:1", "TARGET:1", "BOTH:1", "TARGET:1"],
        }
        quick_test(config_dict)

    def test_natr(self):
        config_dict = {
            'model': 'NATR',
            'train_epochs': ["SOURCE:1", "TARGET:1"],

        }
        quick_test(config_dict)


if __name__ == '__main__':
    unittest.main()

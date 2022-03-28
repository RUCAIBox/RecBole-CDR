# @Time   : 2022/3/11
# @Author : Zihan Lin
# @Email  : zhlin@ruc.edu.cn

import argparse

from recbole_cdr.quick_start import run_recbole_cdr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='CMF', help='name of models')
    parser.add_argument('--config_files', type=str, default=None, help='config files')

    args, _ = parser.parse_known_args()

    config_file_list = args.config_files.strip().split(' ') if args.config_files else None
    run_recbole_cdr(model=args.model, config_file_list=config_file_list)

![RecBole Logo](asset/logo.png)

--------------------------------------------------------------------------------

# RecBole-CrossDomain


[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)


[中文版]

[中文版]: README_CN.md

RecBole-CrossDomain is developed based on RecBole for reproducing and developing cross domain recommendation algorithms.

## Quick-Start
With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole.py
```

This script will run the CMF model with ml-1m as source domain dataset and ml-100k as target domain dataset.


## Cite
If you find RecBole useful for your research or development, please cite the following [paper](https://arxiv.org/abs/2011.01731):

```
@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Zhao, Wayne Xin and Mu, Shanlei and Hou, Yupeng and Lin, Zihan and Chen, Yushuo and Pan, Xingyu and Li, Kaiyuan and Lu, Yujie and Wang, Hui and Tian, Changxin and others},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={4653--4664},
  year={2021}
}
```

## The Team
RecBole-CrossDomain is developed and maintained by [RUC](https://www.recbole.io/about.html).

Here is the list of our lead developers in each development phase.

|           Time           |     Version     |                                                                                          Lead Developers                                                                                          |
|:------------------------:|:---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Mar 2022<br> ~<br> now  |     v0.1.0      | Zihan Lin([@linzihan-backforward](https://github.com/linzihan-backforward)) , Shanlei Mu ([@ShanleiMu](https://github.com/ShanleiMu)) , Gaowei Zhang ([@Wicknight](https://github.com/Wicknight)) |



## License
RecBole-CrossDomain uses [MIT License](./LICENSE).

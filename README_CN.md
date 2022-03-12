![RecBole Logo](asset/logo.png)

--------------------------------------------------------------------------------

# RecBole-CrossDomain (用于跨领域推荐的伯乐)



[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)


[English Version]


[English Version]: README.md


RecBole-CrossDomain 是一个基于 RecBole的 代码库，其针对跨领域推荐算法。


## 快速上手
如果你从GitHub下载了RecBole-CrossDomain的源码，你可以使用提供的脚本进行简单的使用：

```bash
python run_recbole.py
```

这个例子将会以ml-100k为目标域数据集，以ml-1m为源域数据集运行CMF模型的训练和测试。

一般来说，这个例子将花费不到一分钟的时间，我们会得到一些类似下面的输出：



## 引用
如果你觉得RecBole对你的科研工作有帮助，请引用我们的[论文](https://arxiv.org/abs/2011.01731):

```
@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Zhao, Wayne Xin and Mu, Shanlei and Hou, Yupeng and Lin, Zihan and Chen, Yushuo and Pan, Xingyu and Li, Kaiyuan and Lu, Yujie and Wang, Hui and Tian, Changxin and others},
  booktitle={Proceedings of the 30th ACM International Conference on Information \& Knowledge Management},
  pages={4653--4664},
  year={2021}
}
```

## 项目团队
RecBole-CrossDomain由 [中国人民大学](https://www.recbole.io/cn/about.html) 的同学和老师进行开发和维护。 

以下是伯乐项目的首席开发人员名单。他们是伯乐项目的灵魂人物，为伯乐项目的开发作出了重大贡献！

|          时间段           |   版本   |                                                                                   首席开发者                                                                                    |
|:----------------------:|:------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
|  2022年3月<br> ~<br> 现在  | v0.1.0 | 林子涵 ([@linzihan-backforward](https://github.com/linzihan-backforward))， 牟善磊 ([@ShanleiMu](https://github.com/ShanleiMu)), 张高玮 ([@Wicknight](https://github.com/Wicknight)) |


## 免责声明
RecBole-CrossDomain 基于 [MIT License](./LICENSE) 进行开发，本项目的所有数据和代码只能被用于学术目的。

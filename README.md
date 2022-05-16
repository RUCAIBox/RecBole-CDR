![](asset/recbole-cdr-logo.png)

--------------------------------------------------------------------------------

# RecBole-CDR

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)


[中文版]


[中文版]: README_CN.md


**RecBole-CDR** is a library built upon [RecBole](https://github.com/RUCAIBox/RecBole) for reproducing and developing cross-domain-recommendation algorithms.


![](asset/arch.png)

## Highlights

* **Unified data structure for cross-domain-recommendation**:
    Our library designs a unified data structure for cross domain recommendation, including source domain data, target domain data and overlapping data.
* **Free and rich training strategies**:
    Our library provides four basic training modes for cross-domain-recommendation, and supports users to customize and combine them.
* **Extensive cross-domain-recommendation algorithm library**:
    Based on unified data structure and rich training strategies, cross-domain-recommendation algorithms can be easily implemented and compared with others.

## Requirements

```
recbole>=1.0.0
torch>=1.7.0
python>=3.7.0
```

## Quick-Start

With the source code, you can use the provided script for initial usage of our library:

```bash
python run_recbole_cdr.py
```

This script will run the CMF model with ml-1m as source domain dataset and ml-100k as target domain dataset.

If you want to change the models, just run the script by setting additional command parameters:

```bash
python run_recbole_cdr.py --model=[model]
```

## Implemented Models

We list currently supported Cross-Domain-Recommendation models:

* **[CMF](recbole_cdr/model/cross_domain_recommender/cmf.py)** from Singh *et al.*: [Relational Learning via Collective Matrix Factorization](https://dl.acm.org/doi/10.1145/1401890.1401969) (SIGKDD 2008).
* **[DTCDR](recbole_cdr/model/cross_domain_recommender/dtcdr.py)** from Zhu *et al.*: [DTCDR: A Framework for Dual-Target Cross-Domain Recommendation](https://dl.acm.org/doi/10.1145/3357384.3357992) (CIKM 2019).
* **[CoNet](recbole_cdr/model/cross_domain_recommender/conet.py)** from Hu *et al.*: [CoNet: Collaborative Cross Networks for Cross-Domain Recommendation](http://dl.acm.org/doi/10.1145/3269206.3271684) (CIKM 2018).
* **[BiTGCF](recbole_cdr/model/cross_domain_recommender/bitgcf.py)** from Liu *et al.*: [Cross Domain Recommendation via Bi-directional Transfer Graph Collaborative Filtering Networks](https://dl.acm.org/doi/10.1145/3340531.3412012) (CIKM 2020).
* **[CLFM](recbole_cdr/model/cross_domain_recommender/clfm.py)** from Gao *et al.*: [Cross-Domain Recommendation via Cluster-Level Latent Factor Model](http://www.ecmlpkdd2013.org/wp-content/uploads/2013/07/417.pdf) (PKDD 2013).
* **[DeepAPF](recbole_cdr/model/cross_domain_recommender/deepapf.py)** from Yan *et al.*: [DeepAPF: Deep Attentive Probabilistic Factorization for Multi-site Video Recommendation](https://www.ijcai.org/proceedings/2019/0202.pdf) (IJCAI 2019).
* **[NATR](recbole_cdr/model/cross_domain_recommender/natr.py)** from Gao *et al.*: [Cross-domain Recommendation Without Sharing User-relevant Data](https://dl.acm.org/doi/10.1145/3308558.3313538) (WWW 2019).
* **[EMCDR](recbole_cdr/model/cross_domain_recommender/emcdr.py)** from Man *et al.*: [Cross-Domain Recommendation: An Embedding and Mapping Approach](https://www.ijcai.org/proceedings/2017/343) (IJCAI 2017).
* **[SSCDR](recbole_cdr/model/cross_domain_recommender/sscdr.py)** from Kang *et al.*: [Semi-Supervised Learning for Cross-Domain Recommendation to Cold-Start Users](http://dl.acm.org/doi/10.1145/3357384.3357914) (CIKM 2019).
* **[DCDCSR](recbole_cdr/model/cross_domain_recommender/dcdcsr.py)** from Zhu *et al.*: [A Deep Framework for Cross-Domain and Cross-System Recommendations](https://arxiv.org/abs/2009.06215) (IJCAI 2018).



## Result

### Leaderboard

We carefully tune the hyper-parameters of the implemented models on three pairs of source-target datasets and release the corresponding leaderboards for reference:

- Cross-domain-recommendation on [`Amazon`](results/Amazon-Books.md) datasets; 
- Cross-domain-recommendation on [`Book-Crossing`](results/Book-Crossing.md) datasets; 
- Cross-domain-recommendation on [`Douban`](results/Douban.md) datasets; 


## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/RUCAIBox/RecBole-CDR/issues).

We welcome all contributions from bug fixes to new features and extensions.

We expect all contributions discussed in the issue tracker and going through PRs.


## The Team

RecBole-CDR is developed and maintained by members from [RUCAIBox](http://aibox.ruc.edu.cn/), the main developers are Zihan Lin ([@linzihan-backforward](https://github.com/linzihan-backforward)), Gaowei Zhang ([@Wicknight](https://github.com/Wicknight)) and Shanlei Mu ([@ShanleiMu](https://github.com/ShanleiMu)).


## Acknowledgement

The implementation is based on the open-source recommendation library [RecBole](https://github.com/RUCAIBox/RecBole).

Please cite the following paper as the reference if you use our code or processed datasets.

```
@inproceedings{zhao2021recbole,
  title={Recbole: Towards a unified, comprehensive and efficient framework for recommendation algorithms},
  author={Wayne Xin Zhao and Shanlei Mu and Yupeng Hou and Zihan Lin and Kaiyuan Li and Yushuo Chen and Yujie Lu and Hui Wang and Changxin Tian and Xingyu Pan and Yingqian Min and Zhichao Feng and Xinyan Fan and Xu Chen and Pengfei Wang and Wendi Ji and Yaliang Li and Xiaoling Wang and Ji-Rong Wen},
  booktitle={{CIKM}},
  year={2021}
}

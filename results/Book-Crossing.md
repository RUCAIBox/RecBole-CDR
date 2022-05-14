# Experimental setting

**Source domain dataset**: [Book-Crossing](http://www2.informatik.uni-freiburg.de/~cziegler/BX/)

**Target domain dataset**: [Librarything](https://www.librarything.com)

**Evaluation**: all users in target dataset, ratio-based 8:1:1, full sort

**Metrics**: Recall, Precision, NDCG, MRR, Hit

**Topk**: 10, 20, 50

**Properties**:
```yaml
seed: 2022
field_separator: "\t"
item_link_file_path: ./datasets/Book-Crossing_Librarything.link
source_domain:
  dataset: Book-Crossing
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: ISBN
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, ISBN, rating]
  user_inter_num_interval: "[5,inf)"
  item_inter_num_interval: "[0,inf)"
  val_interval:
    rating: "[5,inf)"
  drop_filter_field: True

target_domain:
  dataset: Librarything
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: book_name
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, book_name, rating]
  user_inter_num_interval: "[5,inf)"
  item_inter_num_interval: "[5,inf)"
  val_interval:
    rating: "[5,inf)"
  drop_filter_field: True
# training and evaluation
epochs: 500
train_batch_size: 4096
eval_batch_size: 4096000
valid_metric: NDCG@10
```
For fairness, we restrict users' and items' embedding dimension as following. Please adjust the name of the corresponding args of different models.
```yaml
embedding_size: 64
```

# Dataset Statistics
| Dataset       | #Users | #items | #Interactions | Sparsity |
|---------------|--------|--------|---------------|----------|
| Book-Crossing | 13581  | 153430 | 324049        | 99.98%   | 
| Librarything  | 6783   | 9502   | 379908        | 99.31%   |

Number of Overlapped User: 0

Number of Overlapped Item: 2799

# Evaluation Results

| Method      | Recall@10 | Precesion@10 | NDCG@10 | MRR@10 | Hit@10 |
|-------------|-----------|--------------|---------|--------|--------|
| **CoNet**   | 0.0602    | 0.0283       | 0.05    | 0.0905 | 0.2248 |
| **CLFM**    | 0.0641    | 0.0306       | 0.0541  | 0.0962 | 0.2375 |
| **DTCDR**   | 0.0735    | 0.0331       | 0.0597  | 0.1008 | 0.2492 |
| **DeepAPF** | 0.1014    | 0.0428       | 0.0843  | 0.1418 | 0.314  |
| **BiTGCF**  | 0.1223    | 0.0502       | 0.0969  | 0.1509 | 0.358  |
| **CMF**     | 0.1197    | 0.0511       | 0.1004  | 0.1664 | 0.3637 |
| **EMCDR**   | 0.1311    | 0.0536       | 0.1098  | 0.1802 | 0.3784 |
| **NATR**    |           |              |         |        |        |
| **SSCDR**   |           |              |         |        |        |
| **DCDCSR**  |           |              |         |        |        |

| Method      | Recall@20 | Precesion@20 | NDCG@20 | MRR@20 | Hit@20 |
|-------------|-----------|--------------|---------|--------|--------|
| **CoNet**   | 0.0978    | 0.0233       | 0.062   | 0.0972 | 0.3227 |
| **CLFM**    | 0.1046    | 0.0251       | 0.0671  | 0.1035 | 0.3425 |
| **DTCDR**   | 0.1178    | 0.0264       | 0.0732  | 0.1084 | 0.3599 |
| **DeepAPF** | 0.1563    | 0.0339       | 0.1017  | 0.1499 | 0.4318 |
| **BiTGCF**  | 0.1851    | 0.0399       | 0.117   | 0.1597 | 0.4840 |
| **CMF**     | 0.1782    | 0.0395       | 0.1186  | 0.1742 | 0.4768 |
| **EMCDR**   | 0.1311    | 0.0536       | 0.1098  | 0.1802 | 0.3784 |
| **NATR**    |           |              |         |        |        |
| **SSCDR**   |           |              |         |        |        |
| **DCDCSR**  |           |              |         |        |        |

| Method      | Recall@50 | Precesion@50 | NDCG@50 | MRR@50 | Hit@50 |
|-------------|-----------|--------------|---------|--------|--------|
| **CoNet**   | 0.1781    | 0.0169       | 0.0852  | 0.1021 | 0.4762 |
| **CLFM**    | 0.1017    | 0.0182       | 0.0922  | 0.1087 | 0.5039 |
| **DTCDR**   | 0.2118    | 0.0193       | 0.1002  | 0.1136 | 0.5237 |
| **DeepAPF** | 0.2577    | 0.0233       | 0.1313  | 0.155  | 0.5876 |
| **BiTGCF**  | 0.2917    | 0.0252       | 0.1528  | 0.1736 | 0.6124 |
| **CMF**     | 0.2932    | 0.0266       | 0.1517  | 0.1793 | 0.6332 |
| **EMCDR**   | 0.3049    | 0.027        | 0.1615  | 0.1931 | 0.6471 |
| **NATR**    |           |              |         |        |        |
| **SSCDR**   |           |              |         |        |        |
| **DCDCSR**  |           |              |         |        |        |

# Hyper-parameters

| Method      | Best hyper-parameters                                                                                                                                                  |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CoNet**   | learning_rate=0.005<br/>mlp_hidden_size=[64,32,16,8]<br/>reg_weight=0.01                                                                                               |
| **CLFM**    | learning_rate=0.0005<br/>share_embedding_size=32<br/>alpha=0.5<br/>reg_weight=0.001                                                                                    |
| **DTCDR**   | learning_rate=0.0005<br/>mlp_hidden_size=[64,64]<br/>dropout_prob=0.1<br/>alpha=0.3<br/>base_model=NeuMF                                                               |
| **DeepAPF** | learning_rate=0.001                                                                                                                                                    |
| **BiTGCF**  | learning_rate=0.0005<br/>n_layers=2<br/>concat_way=concat<br/>lambda_source=0.8<br/>lambda_target=0.8<br/>drop_rate=0.1<br/>reg_weight=0.001                           |
| **CMF**     | learning_rate=0.0005<br/>lambda=0.7<br/>gamma=0.1<br/>alpha=0.3                                                                                                        |
| **EMCDR**   | learning_rate=0.001<br/>mapping_function=linear<br/>mlp_hidden_size=[32]<br/>overlap_batch_size=300<br/>reg_weight=0.001<br/>latent_factor_model=BPR<br/>loss_type=BPR |
| **NATR**    |                                                                                                                                                                        |
| **SSCDR**   |                                                                                                                                                                        |
| **DCDCSR**  |                                                                                                                                                                        |
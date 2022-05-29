# Experimental setting

**Source domain dataset**: [Amazon-Books](http://jmcauley.ucsd.edu/data/amazon)

**Target domain dataset**: [Amazon-Movie](http://jmcauley.ucsd.edu/data/amazon)

**Evaluation**: all users in target dataset, ratio-based 8:1:1, full sort

**Metrics**: Recall, Precision, NDCG, MRR, Hit

**Topk**: 10, 20, 50

**Properties**:
```yaml
seed: 2022
field_separator: "\t"
source_domain:
  dataset: AmazonBooks
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[10,inf)"
  item_inter_num_interval: "[10,inf)"
  val_interval:
    rating: "[3,inf)"
  drop_filter_field: True

target_domain:
  dataset: AmazonMov
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[10,inf)"
  item_inter_num_interval: "[10,inf)"
  val_interval:
    rating: "[3,inf)"
  drop_filter_field: True

epochs: 500
train_batch_size: 4096
eval_batch_size: 409600
valid_metric: NDCG@10

```
For fairness, we restrict users' and items' embedding dimension as following. Please adjust the name of the corresponding args of different models.
```yaml
embedding_size: 64
```

# Dataset Statistics
| Dataset      | #Users | #items | #Interactions | Sparsity |
|--------------|--------|--------|---------------|----------|
| Amazon-Books | 135109 | 115172 | 4042382       | 99.97%   |
| Amazon-Movie | 26968  | 18563  | 762957        | 99.85%   |

Number of Overlapped User: 5982

Number of Overlapped Item: 0

# Evaluation Results

| Method      | Recall@10 | Precesion@10 | NDCG@10 | MRR@10 | Hit@10 |
|-------------|-----------|--------------|---------|--------|--------|
| **CoNet**   | 0.027     | 0.0058       | 0.0159  | 0.0183 | 0.0525 |
| **CLFM**    | 0.0271    | 0.0061       | 0.0162  | 0.0191 | 0.0545 |
| **DTCDR**   | 0.0304    | 0.0063       | 0.0176  | 0.0198 | 0.0567 |
| **DeepAPF** | 0.0511    | 0.0102       | 0.0305  | 0.0337 | 0.0899 |
| **BiTGCF**  | 0.0573    | 0.0115       | 0.034   | 0.037  | 0.1005 |
| **CMF**     | 0.0594    | 0.0116       | 0.0352  | 0.0379 | 0.1016 |
| **EMCDR**   | 0.0594    | 0.0116       | 0.0357  | 0.0392 | 0.1014 |
| **NATR**    | 0.028     | 0.0059	      | 0.0159  | 0.0177 | 0.0534 |
| **SSCDR**   | 0.0633	   | 0.0127       | 0.0381  | 0.0414 | 0.1085 |
| **DCDCSR**  | 0.0455	   | 0.0092	      | 0.0271  | 0.0303 | 0.0813 |

| Method      | Recall@20 | Precesion@20 | NDCG@20 | MRR@20  | Hit@20 |
|-------------|-----------|--------------|---------|---------|--------|
| **CoNet**   | 0.0453    | 0.005        | 0.021   | 0.0205  | 0.0841 |
| **CLFM**    | 0.0492    | 0.0054       | 0.0222  | 0.0216  | 0.0912 |
| **DTCDR**   | 0.053     | 0.0056       | 0.0239  | 0.0225  | 0.0949 |
| **DeepAPF** | 0.0818    | 0.0083       | 0.0388  | 0.0369  | 0.1382 |
| **BiTGCF**  | 0.0915    | 0.0095       | 0.0434  | 0.0407  | 0.1537 |
| **CMF**     | 0.0947    | 0.0095       | 0.0449  | 0.0417  | 0.1561 |
| **EMCDR**   | 0.0934    | 0.0093       | 0.0449  | 0.0428  | 0.1526 |
| **NATR**    | 0.047     | 0.005        | 0.0212  | 0.0199  | 0.0857 |
| **SSCDR**   | 0.0997    | 0.0103       | 0.0479  | 0.0452  | 0.164  |
| **DCDCSR**  | 0.074     | 0.0076	      | 0.0349  | 0.0334  | 0.1273 |

| Method      | Recall@50 | Precesion@50 | NDCG@50 | MRR@50  | Hit@50  |
|-------------|-----------|--------------|---------|---------|---------|
| **CoNet**   | 0.0887    | 0.004        | 0.0307  | 0.0225  | 0.1503  |
| **CLFM**    | 0.0963    | 0.0043       | 0.0327  | 0.0238  | 0.1625  |
| **DTCDR**   | 0.1041    | 0.0045       | 0.0353  | 0.0248  | 0.1721  |
| **DeepAPF** | 0.1455    | 0.0062       | 0.0532  | 0.0398  | 0.2294  |
| **BiTGCF**  | 0.165     | 0.0069       | 0.0598  | 0.0438  | 0.2532  |
| **CMF**     | 0.1645    | 0.0069       | 0.0606  | 0.0447  | 0.2516  |
| **EMCDR**   | 0.1595    | 0.0067       | 0.0597  | 0.0457  | 0.246   |
| **NATR**    | 0.0898	   | 0.004        | 0.0308  | 0.022	  | 0.1533  |
| **SSCDR**   | 0.1709	   | 0.0073	      | 0.064   | 0.0483  | 0.2619  |
| **DCDCSR**  | 0.1343	   | 0.0056       | 0.0485  | 0.0361  | 0.2126  |

# Hyper-parameters

| Method      | Best hyper-parameters                                                                                                                                                      |
|-------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CoNet**   | learning_rate=0.005<br/>mlp_hidden_size=[32,32,16,8]<br/>reg_weight=0.001                                                                                                  |
| **CLFM**    | learning_rate=0.0005<br/>share_embedding_size=32<br/>alpha=0.1<br/>reg_weight=0.0001                                                                                       |
| **DTCDR**   | learning_rate=0.0005<br/>mlp_hidden_size=[64,64]<br/>dropout_prob=0.3<br/>alpha=0.3<br/>base_model=NeuMF                                                                   |
| **DeepAPF** | learning_rate=0.00001                                                                                                                                                      |
| **BiTGCF**  | learning_rate=0.0001<br/>n_layers=3<br/>concat_way=mean<br/>lambda_source=0.8<br/>lambda_target=0.8<br/>drop_rate=0.1<br/>reg_weight=0.01                                  |
| **CMF**     | learning_rate=0.0005<br/>lambda=0.2<br/>gamma=0.1<br/>alpha=0.2                                                                                                            |
| **EMCDR**   | learning_rate=0.001<br/>mapping_function=non_linear<br/>mlp_hidden_size=[128]<br/>overlap_batch_size=300<br/>reg_weight=0.01<br/>latent_factor_model=BPR<br/>loss_type=BPR |
| **NATR**    | learning_rate=0.001<br/>max_inter_length=100<br/>reg_weight=1e-5                                                                                                           |
| **SSCDR**   | learning_rate=0.0005<br/>lambda=0.05<br/>margin=0.3<br/>overlap_batch_size=1024                                                                                            |
| **DCDCSR**  | learning_rate=0.0005<br/>mlp_hidden_size=[128]<br/>k=10                                                                                                                    |

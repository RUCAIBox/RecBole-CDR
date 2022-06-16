# Experimental setting

**Source domain dataset**: [Douban-Books](https://www.douban.com/)

**Target domain dataset**: [Douban-Music](https://www.douban.com/)

**Evaluation**: all users in target dataset, ratio-based 8:1:1, full sort

**Metrics**: Recall, Precision, NDCG, MRR, Hit

**Topk**: 10, 20, 50

**Properties**:
```yaml
field_separator: "\t"
source_domain:
  dataset: DoubanBook
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[5,inf)"
  item_inter_num_interval: "[5,inf)"
  val_interval:
    rating: "[3,inf)"
  drop_filter_field: True

target_domain:
  dataset: DoubanMovie
  USER_ID_FIELD: user_id
  ITEM_ID_FIELD: item_id
  RATING_FIELD: rating
  TIME_FIELD: timestamp
  NEG_PREFIX: neg_
  LABEL_FIELD: label
  load_col:
    inter: [user_id, item_id, rating]
  user_inter_num_interval: "[5,inf)"
  item_inter_num_interval: "[5,inf)"
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
| Douban-Book  | 18085  | 33067  | 809248        | 99.86%   |
| Douban-Movie | 22041  | 25802  | 2552305       | 99.55%   |

Number of Overlapped User: 15434

Number of Overlapped Item: 0


# Hyper-parameters

| Method      | Best hyper-parameters                                                                                                                                                     |
|-------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CoNet**   | learning_rate=0.005<br/>mlp_hidden_size=[64,32,16,8]<br/>reg_weight=0.01                                                                                                  |
| **CLFM**    | learning_rate=0.0005<br/>share_embedding_size=48<br/>alpha=0.1<br/>reg_weight=0.0001                                                                                      |
| **DTCDR**   | learning_rate=0.0005<br/>mlp_hidden_size=[64,64]<br/>dropout_prob=0.2<br/>alpha=0.1<br/>base_model=NeuMF                                                                  |
| **DeepAPF** | learning_rate=0.0005                                                                                                                                                      |
| **BiTGCF**  | learning_rate=0.0005<br/>n_layers=2<br/>concat_way=mean<br/>lambda_source=0.8<br/>lambda_target=0.8<br/>drop_rate=0.1<br/>reg_weight=0.01                                 |
| **CMF**     | learning_rate=0.0005<br/>lambda=0.9<br/>gamma=0.1<br/>alpha=0.1                                                                                                           |
| **EMCDR**   | learning_rate=0.001<br/>mapping_function=non_linear<br/>mlp_hidden_size=[64]<br/>overlap_batch_size=100<br/>reg_weight=0.01<br/>latent_factor_model=BPR<br/>loss_type=BPR |
| **NATR**    | learning_rate=0.001<br/>max_inter_length=100<br/>reg_weight=1e-5                                                                                                          |
| **SSCDR**   | learning_rate=0.0005<br/>lambda=0<br/>margin=0.2<br/>overlap_batch_size=1024                                                                                              |
| **DCDCSR**  | learning_rate=0.0005<br/>mlp_hidden_size=[128]<br/>k=10                                                                                                                   |

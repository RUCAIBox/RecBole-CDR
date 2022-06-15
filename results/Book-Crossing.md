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


# Hyper-parameters

| Method      | Best hyper-parameters                                                                                                                                                  |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CoNet**   | learning_rate=0.005<br/>mlp_hidden_size=[64,32,16,8]<br/>reg_weight=0.01                                                                                               |
| **CLFM**    | learning_rate=0.0005<br/>share_embedding_size=32<br/>alpha=0.5<br/>reg_weight=0.001                                                                                    |
| **DTCDR**   | learning_rate=0.0005<br/>mlp_hidden_size=[64,64]<br/>dropout_prob=0.3<br/>alpha=0.3<br/>base_model=NeuMF                                                               |
| **DeepAPF** | learning_rate=0.001                                                                                                                                                    |
| **BiTGCF**  | learning_rate=0.0005<br/>n_layers=2<br/>concat_way=concat<br/>lambda_source=0.8<br/>lambda_target=0.8<br/>drop_rate=0.1<br/>reg_weight=0.001                           |
| **CMF**     | learning_rate=0.0005<br/>lambda=0.7<br/>gamma=0.1<br/>alpha=0.3                                                                                                        |
| **EMCDR**   | learning_rate=0.001<br/>mapping_function=linear<br/>mlp_hidden_size=[32]<br/>overlap_batch_size=300<br/>reg_weight=0.001<br/>latent_factor_model=BPR<br/>loss_type=BPR |
| **NATR**    | learning_rate=0.005<br/>max_inter_length=100<br/>reg_weight=1e-5                                                                                                       |
| **SSCDR**   | learning_rate=0.0005<br/>lambda=0<br/>margin=0.2<br/>overlap_batch_size=1024                                                                                           |
| **DCDCSR**  | learning_rate=0.0005<br/>mlp_hidden_size=[128]<br/>k=10                                                                                                                |

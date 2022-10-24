# Xatu

Anonymous code repository for Xatu.

## Dependencies
- Python 3.8 (Anaconda installation recommended)
- See `environment.yml` for a list of Python library dependencies

## Dataset
- `xatu_train/val/test_all_feats_and_labels.pt`: {'feats_real':(# of datapoints, # of time slots, # of features), 'labels':[label for datapoint 1, label for datapoint 2, ..., label for datapoint n]}. This dataset is for machine learning.
- `xatu_data_val/test.json`: [dict for datapoint 1, dict for datapoint 2, ..., dict for datapoint n], each dict include under_attack_index, anomaly_start, attack_traffic, legit_traffic. This dataset is for evaluating effectiveness and overhead.

## How to run Xatu?
```
$ python run.py

---------- configuration:
namespace(batch_size=0, epochs=10, grad_max_norm=0, lr=0.01, model_class='ScoringModelLSTM', model_config={'hidden_size': 10}, weight_decay=0)

---------- training:
100%|███████████████████████████████████████████████████████████████████████████████| 10/10 [00:00<00:00, ... epochs/s]
model saved at ./saved_models/lstm-10/scoring-model-lstm_traffic-feats-basic3.pt
predicted scores for Xatu val data saved at ./saved_models/lstm-10/scoring-model-lstm_traffic-feats-basic3_scores_val.pt
predicted scores for Xatu testing data saved at ./saved_models/lstm-10/scoring-model-lstm_traffic-feats-basic3_scores_test.pt
--------------------------------------------------

Evaluation to bound overhead on val data and check effectiveness on test data

val data statistics:
number of nodes: ...
number of attacks: ... (total: ...)
overhead 25th ..., 50th ..., 75th ..., 100th ...
effectiveness 10th ..., 50th ..., 90th ...
searched threshold on val data: ...

test data statistics:
number of nodes: ...
number of attacks: ... (total: ...)
overhead 25th ..., 50th ..., 75th ..., 100th ...
effectiveness 10th ..., 50th ..., 90th ...

evaluation results (including searched threshold) for Xatu testing data saved at ./saved_models/lstm-10/scoring-model-lstm_traffic-feats-basic3_results_dict.pt
Number of parameters: 1,831

--------------------------------------------------
ScoringModelLSTM(
  (lstm_1): LSTM(3, 10, batch_first=True)
  ...
  (proj): Linear(in_features=30, out_features=1, bias=True)
)
--------------------------------------------------
{'effectiveness_50': ..., 'overhead_75': ..., ', ...}
````

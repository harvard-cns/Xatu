"""Use the extracted traffic features from the pretrained model and the real traffic features to learn
parameters of a simple feedforward neural network model for Xatu early detection.
For 1351 nodes.

We use the Xatu validation data to train, and test on the Xatu test data.

The loss is from a survival model (SAFE loss).
"""
import argparse
import os
from types import SimpleNamespace

import torch
import jsons
import json

from eval import XatuEvaluator
from train import xatu_train


def parse_args():
    parser = argparse.ArgumentParser(description='fine-tune on Xatu val data with history traffic features')
    parser.add_argument('--hidden_size', type=int, default=10,
                        help='hidden sizes of the feedforward network')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='batch size; default is 0 which is to use full data (no minibatch)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    tag_hidden_size = f'-{args.hidden_size}'

    data_dir = './toy'
    data_name_train = 'xatu_train_all_feats_and_labels.pt'
    data_name_val = 'xatu_val_all_feats_and_labels.pt'
    data_name_test = 'xatu_test_all_feats_and_labels.pt'
    xatu_data_name_val = 'xatu_data_val.json'
    xatu_data_name_test = 'xatu_data_test.json'

    save_dir = f'./saved_models/lstm{tag_hidden_size}'

    # ========== define features to be used and save name
    # select 3 features from original
    num_features = 3
    feature_ids = torch.LongTensor([0, 1, 2])
    save_name = 'scoring-model-lstm_traffic-feats-basic3.pt'

    os.makedirs(save_dir, exist_ok=True)

    # ========== load features and labels
    # index_in_xatu, labels, feats_real, node_ids, feats_disc_time_node
    data_dict_train = torch.load(os.path.join(data_dir, data_name_train))    
    all_feats_train = data_dict_train['feats_real']    # (bsz, time, feature)
    all_labels_train = torch.LongTensor(data_dict_train['labels'])    # (bsz)

    # Xatu val set features
    data_dict_val = torch.load(os.path.join(data_dir, data_name_val))
    all_feats_val = data_dict_val['feats_real']    # (bsz, time, feature)
    
    # Xatu test set features
    data_dict_test = torch.load(os.path.join(data_dir, data_name_test))
    all_feats_test = data_dict_test['feats_real']    # (bsz, time, feature)

    # ========== initialize evaluator
    xatu_data_val = json.load(open(os.path.join(data_dir, xatu_data_name_val), 'r'))
    xatu_data_test = json.load(open(os.path.join(data_dir, xatu_data_name_val), 'r'))
    xatu_evaluator = XatuEvaluator(
        xatu_data_val, xatu_data_test,
    )

    # ========== config model and optimization
    config = {
        'model_class': 'ScoringModelLSTM',
        'model_config': {
            'hidden_size': args.hidden_size,
        },
        'lr': 0.01,
        'weight_decay': 0,
        'grad_max_norm': 0,
        'epochs': 10,
        'batch_size': args.batch_size,
    }

    config = SimpleNamespace(**config)

    print()
    print('-' * 10 + ' configuration:')
    print(config)
    print()
    
    # ========== train and test
    print('-' * 10 + ' training:')
    results_dict, train_loss, model, scores_val, scores_test = xatu_train(
        config,
        save_dir=save_dir,
        num_features=num_features,
        feature_ids=feature_ids,
        save_name=save_name,
        all_feats_train=all_feats_train,
        all_labels_train=all_labels_train,
        all_feats_val=all_feats_val,
        all_feats_test=all_feats_test,
        xatu_evaluator=xatu_evaluator,
        verbose=False,
    )

    # print model
    print()
    print('-' * 50)
    print(model)

    # print result summary
    print('-' * 50)
    print(
        {'effectiveness_50': results_dict['effectiveness_50'],
         'overhead_75': results_dict['overhead_75'],
         'threshold': results_dict['threshold'],
         'train_loss': train_loss,
         'config': jsons.dump(config),
         }
    )
    print()

    # breakpoint()
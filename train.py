import os
import random
from itertools import product
import pickle

import torch
import torch.nn as nn
from tqdm import tqdm
import jsons

from model import SAFELoss
from model import ScoringModelLSTM


def xatu_train(config, *, save_dir, num_features, feature_ids, save_name,
          all_feats_train, all_labels_train, all_feats_val, all_feats_test, xatu_evaluator, verbose=False):
    # ========== parameters
    seed = 0
    devid = 0

    # optimization hyper-parameters
    lr = config.lr
    weight_decay = config.weight_decay
    epochs = 500 if not hasattr(config, 'epochs') else (config.epochs or 500)
    grad_max_norm = config.grad_max_norm

    # ========== random seeds and device
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(f'cuda:{devid}') if devid > -1 else torch.device('cpu')

    # ========== define model and loss
    model = ScoringModelLSTM(num_features=num_features, feature_ids=feature_ids,
                        **config.model_config)

    model.to(device)
    criterion = SAFELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)

    # ========== training the model (no minibatch)
    for ep in tqdm(range(epochs), unit=' epochs'):
        if verbose:
            print(f'ep / epochs: {ep + 1} / {epochs} ' + '-' * 20)
        model.train()

        if (not hasattr(config, 'batch_size')) or config.batch_size == 0:
            # ===== full data (no batch)
            all_feats_train = all_feats_train.to(device)
            optimizer.zero_grad()
            out = model(all_feats_train).squeeze(2)
            # breakpoint()
            loss = criterion(out, all_labels_train.to(device))
            loss.backward()

            # clip gradient
            if grad_max_norm:
                nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)

            # check grad NaN (the loss may be not NaN): if so, exit training loop
            if torch.isnan(sum(p.grad.sum() for p in model.parameters())):
                config.epochs = ep
                break

            optimizer.step()

            if verbose:
                print(f'training loss --- {loss.item():.5f}')

        else:
            # ===== iterate over batches
            batch_size = config.batch_size
            total_loss = 0
            total_examples = 0
            if_break = False

            # split data into batches and shuffle
            all_feats_batches, all_labels_batches = \
                torch.split(all_feats_train, batch_size), torch.split(all_labels_train, batch_size)
            num_batches = len(all_feats_batches)
            # for feats, labels in zip(torch.split(all_feats, batch_size), torch.split(all_labels, batch_size)):
            for batch_id in torch.randperm(num_batches):    # shuffle batches
                feats, labels = all_feats_batches[batch_id], all_labels_batches[batch_id]

                feats = feats.to(device)
                optimizer.zero_grad()
                out = model(feats).squeeze(2)
                loss = criterion(out, labels.to(device))
                loss.backward()

                # clip gradient
                if grad_max_norm:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_max_norm)

                # check grad NaN (the loss may be not NaN): if so, exit training loop
                if torch.isnan(sum(p.grad.sum() for p in model.parameters() if p.grad is not None)):
                    if_break = True
                    break

                optimizer.step()

                # NOTE loss here is the average (vs. sum) in SAFELoss now
                total_loss += loss.clone().detach() * len(labels)
                total_examples += len(labels)

            # check grad NaN (the loss may be not NaN): if so, exit training loop
            if if_break:
                config.epochs = ep
                break

            loss = total_loss / total_examples

            if verbose:
                print(f'training loss --- {loss:.5f}')

    save_path = os.path.join(save_dir, save_name)
    # below doesn't save the fixed parameters
    # torch.save(model.state_dict(), save_path)
    torch.save(model.cpu(), save_path)
    print(f'model saved at {save_path}')

    # ========== dump the prediction results
    model.to(device)    # NOTE in above model saving, model is moved to CPU
    model.eval()

    # Xatu val set
    with torch.no_grad():
        if (not hasattr(config, 'batch_size')) or config.batch_size == 0:
            scores_val = model.scoring(all_feats_val.to(device)).detach().cpu()
        else:
            scores_list = []
            for feats in torch.split(all_feats_val, batch_size):
                scores_list.append(model.scoring(feats.to(device)).detach().cpu())    # (bsz, 30)
            scores_val = torch.cat(scores_list, dim=0)

    save_name_val = save_name.split('.pt')[0] + '_scores_val.pt'
    save_path = os.path.join(save_dir, save_name_val)
    torch.save(scores_val, save_path)
    print(f'predicted scores for Xatu val data saved at {save_path}')

    # Xatu test set
    with torch.no_grad():
        if (not hasattr(config, 'batch_size')) or config.batch_size == 0:
            scores_test = model.scoring(all_feats_test.to(device)).detach().cpu()
        else:
            scores_list = []
            for feats in torch.split(all_feats_test, batch_size):
                scores_list.append(model.scoring(feats.to(device)).detach().cpu())    # (bsz, 30)
            scores_test = torch.cat(scores_list, dim=0)

    save_name_test = save_name.split('.pt')[0] + '_scores_test.pt'
    save_path = os.path.join(save_dir, save_name_test)
    torch.save(scores_test, save_path)
    print(f'predicted scores for Xatu testing data saved at {save_path}')

    # ========== evaluate on test data
    results_dict = xatu_evaluator.evaluate(scores_val.numpy(), scores_test.numpy(), score_type='survival',
                                           only_test=False)
    # save results (including searched threshold)
    save_path = os.path.join(save_dir, save_name.split('.pt')[0] + '_results_dict.pt')
    torch.save(results_dict, save_path)
    print(f'evaluation results (including searched threshold) for Xatu testing data saved at {save_path}')

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters()):,}')

    return results_dict, loss.item(), model, scores_val, scores_test
import torch
import torch.nn as nn


XATU_LENGTH = 30


class ScoringModelLSTM(nn.Module):
    def __init__(self, num_features, feature_ids=None, hidden_size=256):
        super().__init__()

        self.num_features = num_features
        if feature_ids is None:
            # feature_ids = torch.arange(num_features)
            pass
        else:
            assert len(feature_ids) == num_features
        self.feature_ids = feature_ids

        self.lstm_1 = nn.LSTM(num_features, hidden_size, num_layers=1, batch_first=True)
        self.lstm_10 = nn.LSTM(num_features, hidden_size, num_layers=1, batch_first=True)
        self.lstm_60 = nn.LSTM(num_features, hidden_size, num_layers=1, batch_first=True)

        self.pool_10 = nn.AvgPool1d(10, ceil_mode=True)
        self.pool_60 = nn.AvgPool1d(60, ceil_mode=True)

        self.proj = nn.Linear(hidden_size*3, 1, bias=True)

    def forward(self, features):
        # features: size (batch_size, num_time_steps, num_features)
        # we take the whole history for computation but only last 30 time steps for output
        if self.feature_ids is None:
            features_in = features[:, :, :self.num_features]
        else:
            features_in = features[:, :, self.feature_ids]
        out_1, _ = self.lstm_1(features_in)
        out_10, _ = self.lstm_10(self.pool_10(features_in.permute(0,2,1)).permute(0,2,1))
        out_10 = out_10.repeat_interleave(10, dim=1)[:,:out_1.shape[1],:]
        out_60, _ = self.lstm_60(self.pool_60(features_in.permute(0,2,1)).permute(0,2,1))
        out_60 = out_60.repeat_interleave(60, dim=1)[:,:out_1.shape[1],:]
        out = self.proj(torch.cat((out_1, out_10, out_60), dim=-1))
        # out is of size (batch_size, num_time_steps, 1)
        out = torch.sigmoid(out[:, -XATU_LENGTH:, :])    # hazard rate
        # size (bsz, 30, 1)
        return out

    def scoring(self, features):
        # survival scoring: first step that the score is < a threshold is treated as attack
        lambda_k = self.forward(features).squeeze(2)    # size (bsz, 30)
        scores = torch.exp(-torch.cumsum(lambda_k, dim=1))
        return scores


class SAFELoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        assert reduction in ['sum', 'mean', 'none']
        self.reduction = reduction

    def forward(self, scores, labels):
        # scores: (bsz, 30)
        # labels: (bsz,)
        assert scores.dim() == 2
        bsz, time_steps = scores.size()
        assert time_steps == 30
        # breakpoint()
        labels = labels.float()
        sum_lambda_t = labels * scores[:, :15].sum(1) + (1 - labels) * scores[:, :30].sum(1)    # size (bsz,)
        second_term = labels * torch.log(torch.exp(sum_lambda_t) - 1 + 1e-8)
        # NOTE need for numerical stableness. Or can use torch.finfo().eps, which is 1.1920928955078125e-07
        if self.reduction == 'sum':
            loss = (sum_lambda_t - second_term).sum()
        elif self.reduction == 'mean':
            loss = (sum_lambda_t - second_term).mean()
        elif self.reduction == 'none':
            loss = (sum_lambda_t - second_term)
        else:
            raise NotImplementedError

        return loss
"""Xatu evaluation metric: bounded 75% quantile overhead, look at effectiveness median.

Procedure:
- Based on predicted anomaly scores, bound 75% quantile overhead on val data to search for a detection threshold.
- Apply the same threshold on test data, look at the median of the effectiveness.
"""

from itertools import chain
import json
import os
import pickle
from typing import List, Union, Dict, Tuple

import numpy as np
import torch


XATU_LENGTH = 30


def convert_score_to_survival(score_hazard: Union[List[List[float]], np.ndarray]) -> np.ndarray:
    """Convert the hazard rate, predicted by model (after sigmoid so that its range is [0, 1]), to the
    survival score (the probability that the event has not happened yet).

    Args:
        score_hazard (Union[List[List[float]], np.ndarray]): each of length XATU_LENGTH.

    Returns:
        np.ndarray: size (N, XATU_LENGTH), where N: number of time series.
    """
    score = score_hazard
    if isinstance(score, list):
        score = np.array(score)
    assert isinstance(score, np.ndarray)
    score = score.copy()
    # ----------from hazard rate to survival probability -----------
    for col in range(1, XATU_LENGTH):
        score[:, col] = score[:, col - 1] + score[:, col]
    score_survival = np.exp(-score)
    # --------------------------------------------------------------
    return score_survival


def get_detection(*,
                  score: np.ndarray,
                  threshold: float,
                  score_type: str = 'survival',
                  ) -> List[int]:
    """Get the anomaly detection time based on a fixed threshold, on survival scores (non-increasing with time steps).

    Args:
        score_survival (np.ndarray): [description]
        score_type (str, optional): [description]. Defaults to 'survival'.
        threshold (float): [description]

    Returns:
        List[int]: [description]
    """
    assert score_type in ['survival', 'hazard']
    if score_type == 'survival':
        score_survival = score
    else:
        score_survival = convert_score_to_survival(score)

    detection_times = ((score_survival > threshold).sum(axis=1) - 15).tolist()
    return detection_times


def get_xatu_metric(xatu_data: Dict, detection_times: List[int]) -> Tuple[List[float], List[float], List[float]]:
    """[summary]

    Args:
        xatu_data (Dict): E.g. from './toy/xatu_data_val.json'
        detection_times (List[int]): a list of output detection times range from [-15, 15]

    Returns:
        Tuple[List[float], List[float], List[float]]: [description]

    NOTE
        - The returned overhead and effectiveness values are already converted to 100% by multiplying 100.
    """
    victim2traffic_dict = {}
    effectiveness, overhead, mitigation_time = [], [], []
    for xatu_data_dict, xatu in zip(xatu_data, detection_times):
        victim = xatu_data_dict['under_attack_index']
        if victim not in victim2traffic_dict:
            victim2traffic_dict[victim] = [[], [], []]  # legit to scrubber, attack to scrubber, total attack
        victim2traffic_dict[victim][0].append(sum(xatu_data_dict['legit_traffic'][xatu + 15:]))
        victim2traffic_dict[victim][1].append(sum(xatu_data_dict['attack_traffic'][xatu + 15:]))
        victim2traffic_dict[victim][2].append(sum(xatu_data_dict['attack_traffic']))
        if victim2traffic_dict[victim][2][-1] > 0:
            effectiveness.append(100 * victim2traffic_dict[victim][1][-1] /
                                 victim2traffic_dict[victim][2][-1])  # per attack
            mitigation_time.append(xatu - xatu_data_dict['anomaly_start'])  # per attack
    overhead = [100 * sum(victim2traffic_dict[victim][0]) / sum(victim2traffic_dict[victim][2])
                for victim in victim2traffic_dict if sum(victim2traffic_dict[victim][2]) > 0]  # per customer

    return overhead, effectiveness, mitigation_time


def get_threshold(xatu_data: Dict,
                  score_survival: np.ndarray,
                  overhead_percentile=75,
                  overhead_bound=0.1):
    """Search threshold and select the one that bounds the overhead.
    NOTE `overhead_bound` 0.1 means 0.1%.

    Args:
        xatu_data (Dict): [description]
        score_survival (np.ndarray): [description]
        overhead_percentile (int, optional): [description]. Defaults to 75.
        overhead_bound (float, optional): already on % scale. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    for threshold in np.arange(1, 0, -0.001):
        detection_times = get_detection(score=score_survival, threshold=threshold, score_type='survival')
        overhead, effectiveness, mitigation_time = get_xatu_metric(xatu_data, detection_times)
        if np.percentile(overhead, overhead_percentile) < overhead_bound:  # bound 75 percentile to be below 0.1%
            # NOTE for np.percentile, the parameter should be 75 instead of 0.75; different from np.quantile
            break
    return threshold, detection_times, overhead, effectiveness, mitigation_time


def get_effective_overhead_curve(
    xatu_data: Dict,
    score_survival: np.ndarray,
    overhead_percentile=75,
    ):
    """Search threshold and select the one that bounds the overhead.
    NOTE `overhead_bound` 0.1 means 0.1%.

    Args:
        xatu_data (Dict): [description]
        score_survival (np.ndarray): [description]
        overhead_percentile (int, optional): [description]. Defaults to 75.
        overhead_bound (float, optional): already on % scale. Defaults to 0.1.

    Returns:
        [type]: [description]
    """
    effectiveness_list = []
    overhead_list = []
    mitigation_time_list = []
    for threshold in np.arange(1, 0, -0.001):
        detection_times = get_detection(score=score_survival, threshold=threshold, score_type='survival')
        overhead, effectiveness, mitigation_time = get_xatu_metric(xatu_data, detection_times)
        effectiveness_list.append(np.percentile(effectiveness, 50))
        overhead_list.append(np.percentile(overhead, overhead_percentile))
        mitigation_time_list.append(np.mean(mitigation_time))

    return effectiveness_list, overhead_list, mitigation_time_list


def search_on_val_then_test(xatu_data_val: Dict,
                            score_val: np.ndarray,
                            xatu_data_test: Dict,
                            score_test: np.ndarray,
                            score_type: str = 'survival',
                            overhead_percentile=75,
                            overhead_bound=0.1):
    """Search threshold on validation data to bound overhead, and then apply threshold on test data.
    NOTE this requires val and test data are similar. could check the resulted test overhead as well.

    Args:
        xatu_data_val (Dict): [description]
        score_val (np.ndarray): [description]
        xatu_data_test (Dict): [description]
        score_test (np.ndarray): [description]
        score_type (str, optional): [description]. Defaults to 'survival'.
        overhead_percentile (int, optional): [description]. Defaults to 75.
        overhead_bound (float, optional): already on % scale. Defaults to 0.1.
    """
    assert score_type in ['survival', 'hazard']
    if score_type == 'survival':
        score_survival_val = score_val
        score_survival_test = score_test
    else:
        score_survival_val = convert_score_to_survival(score_val)
        score_survival_test = convert_score_to_survival(score_test)

    # search for threshold on val data
    threshold, detection_times, overhead, effectiveness, mitigation_time = get_threshold(
        xatu_data_val, score_survival_val,
        overhead_percentile=overhead_percentile, overhead_bound=overhead_bound
        )
    print('val data statistics:')
    print('number of nodes:', len(overhead))
    print(f'number of attacks: {len(effectiveness)} (total: {len(xatu_data_val)})')
    print(f'overhead 25th {np.percentile(overhead, 25):.3f}, 50th {np.percentile(overhead, 50):.3f}, '
          f'75th {np.percentile(overhead, 75):.3f}, 100th {np.percentile(overhead, 100):.3f}')
    print(f'effectiveness 10th {np.percentile(effectiveness, 10):.3f}, 50th {np.percentile(effectiveness, 50):.3f}, '
          f'90th {np.percentile(effectiveness, 90):.3f}')
    print(f'searched threshold on val data: {threshold:.3f}')

    # breakpoint()

    # apply to test data
    if len(xatu_data_test) == 0:
        # no data for test
        print()
        print('No data in test - no results')
        return
    detection_times = get_detection(score=score_survival_test, threshold=threshold, score_type='survival')
    overhead, effectiveness, mitigation_time = get_xatu_metric(xatu_data_test, detection_times)
    print()
    print('test data statistics:')
    print('number of nodes:', len(overhead))
    print(f'number of attacks: {len(effectiveness)} (total: {len(xatu_data_test)})')
    print(f'overhead 25th {np.percentile(overhead, 25):.3f}, 50th {np.percentile(overhead, 50):.3f}, '
          f'75th {np.percentile(overhead, 75):.3f}, 100th {np.percentile(overhead, 100):.3f}')
    print(f'effectiveness 10th {np.percentile(effectiveness, 10):.3f}, 50th {np.percentile(effectiveness, 50):.3f}, '
          f'90th {np.percentile(effectiveness, 90):.3f}')
    print()

    results_dict = {
        'threshold': threshold,
        'detection_times': detection_times,
        'overhead': overhead,
        'effectiveness': effectiveness,
        'mitigation_time': mitigation_time,
        'overhead_75': np.percentile(overhead, 75),
        'effectiveness_50': np.percentile(effectiveness, 50),
    }
    return results_dict


def search_and_bound(xatu_data: Dict,
                     score: np.ndarray,
                     score_type: str = 'survival',
                     overhead_percentile=75,
                     overhead_bound=0.1):
    """Search threshold, and bound the 75% percentile of overhead, and look at effectiveness median.
    i.e. get the curve on the same data, and pick the point with overhead bound to look at.

    Args:
        xatu_data (Dict): [description]
        score (np.ndarray): [description]
        score_type (str, optional): [description]. Defaults to 'survival'.
        overhead_percentile (int, optional): [description]. Defaults to 75.
        overhead_bound (float, optional): [description]. Defaults to 0.1.
    """
    assert score_type in ['survival', 'hazard']
    if score_type == 'survival':
        score_survival = score
    else:
        score_survival = convert_score_to_survival(score)

    # search for threshold on val data
    threshold, detection_times, overhead, effectiveness, mitigation_time = get_threshold(
        xatu_data, score_survival,
        overhead_percentile=overhead_percentile, overhead_bound=overhead_bound
    )
    print('data statistics:')
    print('number of nodes:', len(overhead))
    print(f'number of attacks: {len(effectiveness)} (total: {len(xatu_data)})')
    print(f'threshold on bounded overhead: {threshold:.3f}')

    print(f'overhead 25th {np.percentile(overhead, 25):.3f}, 50th {np.percentile(overhead, 50):.3f}, '
          f'75th {np.percentile(overhead, 75):.3f}, 100th {np.percentile(overhead, 100):.3f}')
    print(f'effectiveness 10th {np.percentile(effectiveness, 10):.3f}, 50th {np.percentile(effectiveness, 50):.3f}, '
          f'90th {np.percentile(effectiveness, 90):.3f}')
    print()

    results_dict = {
        'threshold': threshold,
        'detection_times': detection_times,
        'overhead': overhead,
        'effectiveness': effectiveness,
        'mitigation_time': mitigation_time,
        'overhead_75': np.percentile(overhead, 75),
        'effectiveness_50': np.percentile(effectiveness, 50),
    }
    return results_dict
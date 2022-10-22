import json
import os

from eval_utils import convert_score_to_survival, search_on_val_then_test, search_and_bound, get_effective_overhead_curve

class XatuEvaluator:

    def __init__(self, xatu_data_val, xatu_data_test,
                 overhead_percentile=75, overhead_bound=0.1):
        # original Xatu data
        self.xatu_data_val = xatu_data_val
        self.xatu_data_test = xatu_data_test

        self.overhead_percentile = overhead_percentile
        self.overhead_bound = overhead_bound

    def evaluate(self, score_val, score_test, score_type='survival', only_test=True):
        assert score_type in ['survival', 'hazard']
        if score_type == 'hazard':
            if score_val is not None:
                # NOTE `score_val` is not needed when `only_test=True`
                score_val = convert_score_to_survival(score_val)
            score_test = convert_score_to_survival(score_test)

        # ========== search on val data and bound on test data ==========
        if not only_test:
            print('-' * 50)
            print()
            print('Evaluation to bound overhead on val data and check effectiveness on test data')
            print()

            # look at all attacks, regardless of type
            results_dict = search_on_val_then_test(
                xatu_data_val=self.xatu_data_val,
                score_val=score_val,
                xatu_data_test=self.xatu_data_test,
                score_test=score_test,
                score_type='survival',
                overhead_percentile=self.overhead_percentile,
                overhead_bound=self.overhead_bound,
            )

        else:
            # ========== search and bound on the same data ==========
            print('-' * 50)
            print()
            print('Evaluation to look at the effectiveness - overhead curve on the same data')
            print()

            # look at all attacks, regardless of type
            results_dict = search_and_bound(
                xatu_data=self.xatu_data_test,
                score=score_test,
                score_type='survival',
                overhead_percentile=self.overhead_percentile,
                overhead_bound=self.overhead_bound,
            )

        # get the effective - overhead curve
        effectiveness_list, overhead_list, mitigation_time_list = get_effective_overhead_curve(
            self.xatu_data_test, score_test)

        results_dict['eocurve_effectiveness'] = effectiveness_list
        results_dict['eocurve_overhead'] = overhead_list
        results_dict['eocurve_mitigation_time'] = mitigation_time_list

        return results_dict